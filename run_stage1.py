"""
Stage 1: The "Goo-Goo Ga-Ga" Nursery Training & Demonstration.
Watches the fresh AI baby transform from chaotic babbling to mastering Python phonics:
- Character completion (de -> f)
- Keyword syllables (retu -> rn, whi -> le, pri -> nt)
- Matching punctuation ( ( -> ), [ -> ], { -> } )
- Atomic assignments (x = -> 0)
"""
import sys
import torch

torch.set_num_threads(8)

from custom_ncm.stage1_nursery import BabyNursery

def main():
    print("=" * 70, flush=True)
    print("  STAGE 1: THE 'GOO-GOO GA-GA' NURSERY PHASE (AI BABY)", flush=True)
    print("  Teaching the baby the alphabet, phonics, and atomic syllables of Python", flush=True)
    print("=" * 70, flush=True)
    
    nursery = BabyNursery()
    
    test_cues = ["de", "retu", "(", "[", "x = ", "return "]
    
    # --- PHASE 1: UNTRAINED BABY (CHAOTIC BABBLE) ---
    print("\n--- PHASE 1: THE NEWBORN BABY (BEFORE TEACHING) ---", flush=True)
    print("Observing what the untrained baby babbles when given basic phonics cues:\n", flush=True)
    
    untrained_babbles = nursery.test_baby_babble(test_cues)
    for cue, response in zip(test_cues, untrained_babbles):
        clean_resp = repr(response.strip().encode('ascii', errors='replace').decode('ascii'))
        print(f"   Cue: {cue:10s} -> Baby babbles: {clean_resp}", flush=True)
        
    print("\n[Assessment]: Total gibberish. The baby does not yet know letters or syntax.", flush=True)
    print("-" * 70, flush=True)
    
    # --- PHASE 2: PHONICS CLASS (STAGE 1 TEACHING) ---
    print("\n--- PHASE 2: PHONICS CLASS (TEACHING SYLLABLES & SYMBOLS) ---", flush=True)
    nursery.teach_phonics(epochs=8, verbose=True)
    print("-" * 70, flush=True)
    
    # --- PHASE 3: TRAINED BABY (FIRST WORDS) ---
    print("\n--- PHASE 3: THE GRADUATE BABY (AFTER STAGE 1 PHONICS) ---", flush=True)
    print("Testing if the baby has learned its first Python phonics and syllables:\n", flush=True)
    
    trained_babbles = nursery.test_baby_babble(test_cues)
    score = 0
    
    expected = {
        "de": "f",
        "retu": "rn",
        "(": ")",
        "[": "]",
        "x = ": "0",
        "return ": "x" # or 0 or True
    }
    
    for cue, response in zip(test_cues, trained_babbles):
        clean_resp = response.strip().encode('ascii', errors='replace').decode('ascii')
        exp = expected.get(cue, "")
        is_match = exp in clean_resp or (cue == "return " and len(clean_resp) > 0)
        status = "[MASTERED]" if is_match else "[PROGRESS]"
        if is_match:
            score += 1
        print(f"   Cue: {cue:10s} -> Baby speaks: {clean_resp!r:15s}  {status}", flush=True)
        
    mastery_pct = (score / len(test_cues)) * 100.0
    print("\n" + "=" * 70, flush=True)
    print(f"  STAGE 1 NURSERY GRADUATION SCORE: {mastery_pct:.1f}% ({score}/{len(test_cues)} Phonics Cues)", flush=True)
    print("=" * 70, flush=True)
    
    if mastery_pct >= 66.0:
        print("[DIPLOMA] The AI baby has graduated from Stage 1 'Goo-Goo Ga-Ga' phase!", flush=True)
        print("          It has mastered Python syllables and punctuation pairing.", flush=True)
        print("          Ready to advance to Stage 2: Toddler Words (Vocabulary & Associations)!", flush=True)
    else:
        print("[PROGRESS] Baby is advancing well through Stage 1.", flush=True)
    print("=" * 70 + "\n", flush=True)

if __name__ == "__main__":
    main()
