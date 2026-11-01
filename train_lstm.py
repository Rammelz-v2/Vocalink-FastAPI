"""
Train the AAC next-icon LSTM and write lstm_weights.npz.

Training data:
  1) Real sentences students sent from the AAC board (aac_logs.icon_sequence),
     read from DATABASE_URL (or ./vocalink.db by default).
  2) A hand-written seed set of common classroom AAC sentences, so the model
     is usable before enough real data exists. Real sentences are repeated
     REAL_WEIGHT times so they dominate once collected.

Usage:
  python train_lstm.py                 # seed + database
  python train_lstm.py --no-db         # seed only
  python train_lstm.py --epochs 60 --out lstm_weights.npz

Restart the API afterwards (or redeploy) so it loads the new weights.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from lstm_model import NextIconLSTM

REAL_WEIGHT = 3

# Icons that can appear inside a built sentence. Request icons (medicine,
# toilet, emergency, hurt, ...) are excluded: the board sends them as
# requests immediately instead of adding them to the sentence.
ICON_IDS = [
    "water", "food", "rest", "bag", "cold", "hot", "too_loud", "pencil", "tissue",
    "need_break", "go_outside", "fix_chair",
    "happy", "sad", "sick", "scared", "angry", "confused", "excited", "tired", "bored",
    "worried", "frustrated", "calm", "proud", "embarrassed",
    "help", "question", "done", "repeat", "teacher", "understand", "not_understand",
    "more_time", "write", "read", "group_work", "worksheet", "computer", "show_me",
    "notebook", "sit_down", "line_up", "activity",
    "yes", "no", "please", "thankyou", "wait", "stop", "more_a", "give_me", "come_here",
    "open_it", "finished", "go_a", "try_it", "again",
    "math_plus", "math_minus", "math_times", "math_divide", "math_equals", "math_numbers",
    "math_count", "math_shapes", "math_measure", "math_more", "math_less", "math_answer",
    "math_time", "math_money", "math_pattern", "math_bigger", "math_smaller", "math_hundred",
    "num_0", "num_1", "num_2", "num_3", "num_4", "num_5", "num_6", "num_7", "num_8",
    "num_9", "num_10",
    "comm_hello", "comm_sorry", "comm_excuse", "comm_goodjob", "comm_myturn",
    "comm_yourturn", "comm_talk", "comm_listen", "comm_louder", "comm_slower",
    "comm_goodbye", "comm_good_morning", "comm_i_want", "comm_i_like", "comm_i_dont_like",
    "comm_friend", "comm_agree", "comm_disagree",
    "people_me", "people_teacher", "people_friend", "people_classmate", "people_mom",
    "people_dad", "people_nurse", "people_principal", "people_aide",
    "places_classroom", "places_bathroom", "places_cafeteria", "places_library", "places_gym",
    "places_playground", "places_office", "places_hallway", "places_home",
    "flag_au", "flag_br", "flag_ca", "flag_cn", "flag_de", "flag_eg", "flag_es", "flag_fr",
    "flag_gb", "flag_id", "flag_in", "flag_it", "flag_jp", "flag_kr", "flag_mx", "flag_my",
    "flag_ng", "flag_ph", "flag_ru", "flag_sg", "flag_th", "flag_us", "flag_vn", "flag_za",
]
FLAG_IDS = [i for i in ICON_IDS if i.startswith("flag_")]

# Hand-written classroom sentences (icon ids in tap order).
SEED_SENTENCES = [
    "comm_good_morning teacher", "comm_hello teacher", "comm_hello comm_friend",
    "comm_goodbye teacher", "comm_goodbye comm_friend", "comm_good_morning comm_friend",
    "comm_excuse teacher question", "comm_excuse teacher help please",
    "teacher help please", "help please", "help please teacher", "teacher question",
    "question please", "teacher come_here please", "come_here please",
    "not_understand repeat please", "not_understand show_me please", "confused help please",
    "confused question teacher", "repeat please", "repeat comm_slower please",
    "comm_slower please", "comm_louder please", "comm_talk teacher", "show_me again please",
    "understand thankyou", "understand yes", "yes understand", "no not_understand",
    "more_time please", "wait please", "wait more_time please", "done teacher",
    "finished teacher", "done worksheet", "finished worksheet teacher", "done read",
    "comm_i_want water please", "comm_i_want food please", "water please",
    "water please thankyou", "food please", "food please thankyou",
    "comm_i_want rest please", "rest please", "need_break please", "comm_i_want need_break please",
    "tired need_break please", "tired rest please", "go_outside please",
    "comm_i_want go_outside please", "cold teacher", "hot water please", "hot go_outside please",
    "too_loud stop please", "too_loud comm_talk teacher", "pencil please", "give_me pencil please",
    "comm_i_want pencil please", "tissue please", "give_me tissue please", "sick tissue please",
    "fix_chair please", "fix_chair teacher help", "bag please", "give_me bag please",
    "open_it bag please", "open_it please help", "give_me notebook please", "notebook please",
    "give_me worksheet please", "worksheet please", "computer please", "comm_i_want computer please",
    "comm_i_like computer", "comm_i_like read", "comm_i_like activity", "comm_i_like group_work",
    "comm_i_dont_like group_work", "comm_i_dont_like activity", "comm_i_like comm_friend",
    "happy thankyou", "happy teacher", "happy comm_friend", "excited activity",
    "excited go_outside", "proud done", "proud math_answer", "calm thankyou",
    "sad teacher", "sad help please", "sad comm_talk teacher", "scared teacher help",
    "scared help please", "worried question teacher", "worried comm_talk teacher",
    "angry stop", "angry need_break please", "frustrated help please", "frustrated need_break",
    "frustrated not_understand help", "embarrassed comm_talk teacher", "bored activity please",
    "bored comm_i_want computer", "sick teacher", "sick rest please", "sick help teacher",
    "tired teacher", "stop please", "stop no", "no thankyou", "yes please", "yes thankyou",
    "thankyou teacher", "comm_sorry teacher", "comm_sorry comm_friend", "comm_goodjob comm_friend",
    "comm_myturn please", "comm_yourturn", "comm_myturn read", "comm_myturn try_it",
    "try_it again", "try_it please", "again please", "more_a please", "more_a water please",
    "give_me more_a please", "comm_agree", "comm_agree yes", "comm_disagree no",
    "comm_listen please", "comm_listen teacher", "sit_down please", "line_up go_a",
    "go_a go_outside", "group_work comm_friend", "comm_i_want group_work comm_friend",
    "write notebook", "read again", "read please", "write please help", "activity done",
    "question math_answer", "math_answer num_5", "math_answer yes", "math_answer question",
    "math_numbers help", "math_count help please", "math_shapes question",
    "math_measure help", "math_time question", "math_money question", "math_pattern help",
    "math_bigger math_smaller question", "math_more math_less question", "math_hundred",
    "not_understand math_numbers", "help math_count please", "show_me math_shapes please",
    # People
    "people_me sick", "people_me tired", "people_me happy", "people_me sad", "people_me done",
    "people_me comm_i_want water please", "people_me need_break please",
    "comm_i_want people_mom", "comm_i_want people_dad", "comm_i_want people_mom please",
    "people_mom come_here please", "comm_talk people_mom please", "comm_talk people_dad please",
    "sick people_nurse please", "comm_i_want people_nurse please", "people_me sick people_nurse",
    "go_a places_office people_nurse", "people_teacher help please", "people_teacher question",
    "comm_hello people_teacher", "comm_good_morning people_teacher", "thankyou people_teacher",
    "comm_hello people_principal", "comm_talk people_principal", "people_aide help please",
    "people_aide come_here please", "comm_hello people_friend", "comm_i_like people_friend",
    "group_work people_classmate", "comm_sorry people_classmate", "people_classmate stop please",
    "comm_myturn people_classmate", "comm_goodjob people_classmate", "happy people_friend",
    # Places
    "go_a places_bathroom please", "comm_i_want go_a places_bathroom please",
    "go_a places_cafeteria", "food places_cafeteria", "comm_i_want go_a places_cafeteria",
    "go_a places_library please", "read places_library", "comm_i_like places_library",
    "go_a places_gym", "excited places_gym", "comm_i_like places_gym",
    "go_a places_playground please", "go_outside places_playground", "happy places_playground",
    "go_a places_office please", "go_a places_classroom", "line_up go_a places_hallway",
    "go_a places_home", "comm_i_want go_a places_home", "sick go_a places_home please",
    "tired go_a places_home", "people_mom places_home", "places_classroom too_loud",
    # Countries
    "people_me flag_ph", "comm_i_like flag_jp", "comm_i_like flag_kr", "people_teacher flag_us",
    "read flag_gb", "question flag_cn", "show_me flag_fr please", "people_mom flag_in",
    "people_dad flag_mx", "comm_i_want go_a flag_au", "comm_i_like flag_br",
]


def _country_sentences() -> list[list[str]]:
    """One or two sentences per flag so every country is a known token."""
    frames = [["people_me"], ["comm_i_like"], ["comm_i_want", "go_a"], ["question"], ["people_friend"]]
    out = []
    for i, flag in enumerate(FLAG_IDS):
        out.append(frames[i % len(frames)] + [flag])
        out.append(frames[(i + 2) % len(frames)] + [flag])
    return out


def _math_sentences(rng: random.Random, n: int = 60) -> list[list[str]]:
    """Simple arithmetic sentences: num op num equals num."""
    ops = {"math_plus": lambda a, b: a + b, "math_minus": lambda a, b: a - b,
           "math_times": lambda a, b: a * b, "math_divide": lambda a, b: a // b if b else None}
    out = []
    while len(out) < n:
        op = rng.choice(list(ops))
        a, b = rng.randint(0, 10), rng.randint(1, 10)
        r = ops[op](a, b)
        if r is None or not 0 <= r <= 10 or (op == "math_divide" and a % b):
            continue
        seq = [f"num_{a}", op, f"num_{b}", "math_equals", f"num_{r}"]
        if rng.random() < 0.1:
            seq = ["math_answer"] + seq
        out.append(seq)
    return out


def seed_corpus(seed: int = 0) -> list[list[str]]:
    rng = random.Random(seed)
    return [s.split() for s in SEED_SENTENCES] * 4 + _country_sentences() * 2 + _math_sentences(rng)


def db_sentences() -> list[list[str]]:
    """Sentences from aac_logs.icon_sequence (one JSON list per sent message)."""
    from sqlalchemy import create_engine, text

    url = os.getenv("DATABASE_URL", "sqlite:///./vocalink.db")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    try:
        with create_engine(url).connect() as conn:
            rows = conn.execute(text(
                "SELECT icon_sequence FROM aac_logs WHERE icon_sequence IS NOT NULL"
            )).fetchall()
    except Exception as e:
        print(f"Could not read aac_logs.icon_sequence ({e.__class__.__name__}); using seed data only")
        return []
    known = set(ICON_IDS)
    out = []
    for (raw,) in rows:
        try:
            seq = [t for t in json.loads(raw) if t in known]
        except (TypeError, ValueError):
            continue
        if seq:
            out.append(seq)
    return out


def _top3_accuracy(predict, sentences):
    hits = total = 0
    for seq in sentences:
        for t in range(1, len(seq)):
            total += 1
            hits += seq[t] in predict(seq[:t])
    return hits / max(total, 1)


def _bigram_baseline(train):
    counts = defaultdict(Counter)
    for seq in train:
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
    return lambda prefix: [w for w, _ in counts[prefix[-1]].most_common(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_weights.npz"))
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    real = [] if args.no_db else db_sentences()
    corpus = seed_corpus() + real * REAL_WEIGHT
    print(f"{len(real)} real sentences from DB, {len(corpus)} training sequences total")

    # Held-out check: train on 85%, report top-3 next-icon accuracy vs. a bigram.
    rng = random.Random(1)
    unique = list({tuple(s) for s in corpus})
    rng.shuffle(unique)
    cut = int(len(unique) * 0.85)
    train_set = [list(s) for s in unique[:cut]]
    test_set = [list(s) for s in unique[cut:]]
    probe = NextIconLSTM(ICON_IDS)
    probe.train(train_set, epochs=args.epochs, verbose=False)
    lstm_acc = _top3_accuracy(lambda p: [w for w, _ in probe.predict(p, 3)], test_set)
    bigram_acc = _top3_accuracy(_bigram_baseline(train_set), test_set)
    print(f"held-out top-3 accuracy: LSTM {lstm_acc:.1%}  |  bigram baseline {bigram_acc:.1%}")

    # Final model on everything.
    model = NextIconLSTM(ICON_IDS)
    model.train(corpus, epochs=args.epochs)
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
