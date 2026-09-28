"""Analyse d'un chant polyphonique : transcription multi-hauteurs -> voix séparées.

Usage : python analyse_voix.py <audio.wav> <dossier_sortie> [--voix N]

Sorties : data.json (notes par voix, accords, tempo, tonalité), MIDI global + par voix,
MusicXML (partition). Transcription : Spotify basic-pitch (modèle ONNX).
"""
import argparse
import json
import math
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

NOMS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PUPITRES = ["Voix 1 (aigu)"] + [f"Voix {i}" for i in range(2, 13)]


def nom_note(p):
    return f"{NOMS[p % 12]}{p // 12 - 1}"


def transcrire(wav):
    onnx = Path(ICASSP_2022_MODEL_PATH).with_name("nmp.onnx")
    _, _, events = predict(
        str(wav), str(onnx),
        onset_threshold=0.5, frame_threshold=0.3,
        minimum_note_length=90, minimum_frequency=65, maximum_frequency=1100,
        multiple_pitch_bends=False, melodia_trick=True,
    )
    notes = [dict(s=float(s), e=float(e), p=int(p), v=float(a)) for s, e, p, a, _ in events]
    notes.sort(key=lambda n: (n["s"], -n["p"]))
    return notes


def nb_voix(notes, duree):
    """Polyphonie typique = 90e centile du nombre de notes simultanées."""
    t = np.arange(0, duree, 0.05)
    poly = np.array([sum(1 for n in notes if n["s"] <= x < n["e"]) for x in t])
    actifs = poly[poly > 0]
    return int(np.clip(np.percentile(actifs, 90), 1, 12)) if len(actifs) else 1, poly, t


def assigner_voix(notes, k):
    """Répartit les notes en k voix en préservant l'ordre des registres et en
    minimisant les sauts (appariement monotone par programmation dynamique)."""
    ps = sorted(n["p"] for n in notes)
    # hauteur de départ de chaque voix : quantiles du registre (voix 0 = la plus aiguë)
    centre = [ps[int((1 - (i + 0.5) / k) * (len(ps) - 1))] for i in range(k)]
    derniere = list(centre)
    fin = [-1.0] * k

    def cout_note(p, v, t0):
        c = abs(p - derniere[v]) + 0.5 * abs(p - centre[v])
        for w in range(k):  # pas de croisement avec les voix qui sonnent encore
            if fin[w] > t0 + 0.04 and ((w < v and p > derniere[w]) or (w > v and p < derniere[w])):
                c += 12
        return c

    i = 0
    while i < len(notes):
        groupe = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j]["s"] - notes[i]["s"] < 0.06:
            groupe.append(notes[j]); j += 1
        t0 = notes[i]["s"]
        groupe.sort(key=lambda n: -n["p"])
        libres = [v for v in range(k) if fin[v] <= t0 + 0.04]
        if len(libres) < len(groupe):  # plus de notes que de voix libres : on coupe la plus ancienne
            libres = sorted(range(k), key=lambda v: fin[v])[:len(groupe)]
            libres.sort()
        # DP : notes (aiguës -> graves) vers voix libres (ordre 0..k-1), monotone
        m, L = len(groupe), len(libres)
        INF = 1e9
        cout = np.full((m + 1, L + 1), INF); cout[0, :] = 0
        choix = np.zeros((m + 1, L + 1), dtype=bool)
        for a in range(1, m + 1):
            for b in range(1, L + 1):
                saut = cout[a, b - 1]  # voix b laissée sans note
                prend = cout[a - 1, b - 1] + cout_note(groupe[a - 1]["p"], libres[b - 1], t0)
                if prend <= saut:
                    cout[a, b], choix[a, b] = prend, True
                else:
                    cout[a, b] = saut
        a, b = m, L
        while a > 0:
            if choix[a, b]:
                v = libres[b - 1]
                groupe[a - 1]["voix"] = v
                derniere[v] = groupe[a - 1]["p"]
                fin[v] = groupe[a - 1]["e"]
                a -= 1
            b -= 1
        i = j
    return notes


def accords(notes, battements, duree):
    """Accord par temps : profil de classes de hauteurs pondéré -> gabarits d'accords."""
    gabarits = {
        "": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11],
        "m7": [0, 3, 7, 10], "m7b5": [0, 3, 6, 10], "dim7": [0, 3, 6, 9], "6": [0, 4, 7, 9],
        "m6": [0, 3, 7, 9], "9": [0, 4, 7, 10, 2], "maj9": [0, 4, 7, 11, 2], "m9": [0, 3, 7, 10, 2],
        "sus4": [0, 5, 7], "7sus4": [0, 5, 7, 10], "13": [0, 4, 10, 2, 9], "aug": [0, 4, 8],
        "7#9": [0, 4, 7, 10, 3], "7b9": [0, 4, 7, 10, 1], "6/9": [0, 4, 7, 9, 2],
    }
    bornes = list(battements) + [duree]
    res = []
    for t0, t1 in zip(bornes[:-1], bornes[1:]):
        prof = np.zeros(12); basse = None
        for n in notes:
            chev = min(n["e"], t1) - max(n["s"], t0)
            if chev > 0:
                prof[n["p"] % 12] += chev * n["v"]
                if basse is None or n["p"] < basse:
                    basse = n["p"]
        if prof.sum() == 0:
            res.append(dict(t=float(t0), nom="—", notes=[])); continue
        prof /= prof.max()
        meilleur = (-1e9, "")
        for r in range(12):
            for q, iv in gabarits.items():
                masque = np.zeros(12); masque[[(r + x) % 12 for x in iv]] = 1
                score = (prof * masque).sum() - 0.6 * (prof * (1 - masque)).sum() - 0.12 * len(iv)
                if basse is not None and basse % 12 == r:
                    score += 0.25
                if score > meilleur[0]:
                    meilleur = (score, NOMS[r] + q, r)
        nom = meilleur[1]
        if basse is not None and basse % 12 != meilleur[2]:
            nom += "/" + NOMS[basse % 12]
        res.append(dict(t=float(t0), nom=nom,
                        notes=[NOMS[i] for i in np.argsort(-prof)[:6] if prof[i] > 0.15]))
    # fusion des temps consécutifs identiques
    fus = []
    for c in res:
        if fus and fus[-1]["nom"] == c["nom"]:
            continue
        fus.append(c)
    return fus


def tonalite(notes):
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mino = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    prof = np.zeros(12)
    for n in notes:
        prof[n["p"] % 12] += (n["e"] - n["s"]) * n["v"]
    best = max(((np.corrcoef(prof, np.roll(g, r))[0, 1], NOMS[r] + (" majeur" if g is maj else " mineur"))
                for r in range(12) for g in (maj, mino)))
    return best[1]


def ecrire_midi(notes, k, tempo, dossier):
    tout = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for v in range(k):
        seul = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        for pm in (tout, seul):
            inst = pretty_midi.Instrument(program=52, name=PUPITRES[v])  # Choir Aahs
            inst.notes = [pretty_midi.Note(velocity=int(40 + 87 * min(1, n["v"])), pitch=n["p"],
                                           start=n["s"], end=n["e"]) for n in notes if n["voix"] == v]
            pm.instruments.append(inst)
        seul.write(str(dossier / f"voix{v + 1}.mid"))
    tout.write(str(dossier / "toutes_voix.mid"))


def ecrire_musicxml(dossier, k, tempo):
    from music21 import converter, clef, tempo as m21tempo
    sc = converter.parse(str(dossier / "toutes_voix.mid"), quarterLengthDivisors=(4, 3))
    for i, part in enumerate(sc.parts):
        part.partName = PUPITRES[i]
        notes_p = [n.pitch.midi for n in part.recurse().notes if n.isNote]
        if notes_p and np.median(notes_p) < 57:
            part.insert(0, clef.BassClef())
        elif notes_p and np.median(notes_p) < 64:
            part.insert(0, clef.Treble8vbClef())
    sc.write("musicxml", fp=str(dossier / "partition.musicxml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav"); ap.add_argument("sortie"); ap.add_argument("--voix", type=int)
    a = ap.parse_args()
    dossier = Path(a.sortie); dossier.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(a.wav, sr=22050, mono=True)
    duree = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])

    notes = transcrire(a.wav)
    k_auto, poly, t = nb_voix(notes, duree)
    k = a.voix or k_auto
    PUPITRES[k - 1] += " (basse)"
    notes = assigner_voix(notes, k)

    voix = []
    for v in range(k):
        nv = [n for n in notes if n["voix"] == v]
        if not nv:
            continue
        ps = [n["p"] for n in nv]
        voix.append(dict(id=v, nom=PUPITRES[v], nb=len(nv), bas=nom_note(min(ps)),
                         haut=nom_note(max(ps)), mediane=nom_note(int(np.median(ps))),
                         occupation=round(sum(n["e"] - n["s"] for n in nv) / duree, 3)))

    data = dict(
        duree=round(duree, 3), tempo=round(tempo, 1), tonalite=tonalite(notes), nb_voix=k,
        nb_voix_auto=k_auto, polyphonie_max=int(poly.max()),
        polyphonie_moy=round(float(poly[poly > 0].mean()), 2),
        battements=[round(float(b), 3) for b in beats],
        voix=voix, accords=accords(notes, beats, duree),
        notes=[dict(s=round(n["s"], 3), e=round(n["e"], 3), p=n["p"], v=round(n["v"], 3),
                    voix=n["voix"], nom=nom_note(n["p"])) for n in notes],
    )
    (dossier / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    ecrire_midi(notes, k, tempo, dossier)
    try:
        ecrire_musicxml(dossier, k, tempo)
    except Exception as ex:  # la partition est un bonus : ne pas bloquer le reste
        print("MusicXML échoué :", ex)
    print(json.dumps({x: data[x] for x in ("duree", "tempo", "tonalite", "nb_voix", "nb_voix_auto",
                                           "polyphonie_max", "polyphonie_moy", "voix")},
                     ensure_ascii=False, indent=1))
    print(len(data["notes"]), "notes,", len(data["accords"]), "accords")


if __name__ == "__main__":
    main()

