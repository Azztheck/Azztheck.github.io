"""Publie une analyse du Transcripteur musical (Z:\\claude projects\\transcripteur-musical) sur le site.

Les notes viennent telles quelles du moteur « toutes les voix » du transcripteur (clé `poly` de
vocals_notes.json). On ajoute seulement : répartition en N voix (pour l'affichage / solo), accords, tempo,
MIDI par voix et MusicXML.

Usage : python publier_transcripteur.py <dossier cache du morceau> <dossier de sortie> --voix N
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np

import analyse_voix as av


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("sortie"); ap.add_argument("--voix", type=int)
    a = ap.parse_args()
    cache, dossier = Path(a.cache), Path(a.sortie)
    dossier.mkdir(parents=True, exist_ok=True)

    src = json.loads((cache / "vocals_notes.json").read_text("utf-8"))
    notes = [dict(s=n["start"], e=n["end"], p=int(n["midi"]), v=0.8) for n in src["poly"]]
    notes.sort(key=lambda n: (n["s"], -n["p"]))

    y, sr = librosa.load(str(cache / "mix.wav"), sr=22050, mono=True)
    duree = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])

    k_auto, poly, _ = av.nb_voix(notes, duree)
    k = a.voix or k_auto
    av.PUPITRES[k - 1] += " (basse)"
    notes = av.assigner_voix(notes, k)

    voix = []
    for v in range(k):
        nv = [n for n in notes if n["voix"] == v]
        if nv:
            ps = [n["p"] for n in nv]
            voix.append(dict(id=v, nom=av.PUPITRES[v], nb=len(nv), bas=av.nom_note(min(ps)),
                             haut=av.nom_note(max(ps)), mediane=av.nom_note(int(np.median(ps))),
                             occupation=round(sum(n["e"] - n["s"] for n in nv) / duree, 3)))
    melodie = [dict(s=round(n["start"], 3), e=round(n["end"], 3), p=int(n["midi"])) for n in src.get("notes", [])
               if "midi" in n]

    data = dict(
        moteur=src.get("poly_engine", "")[:40], sources=src.get("poly_sources"), filtre=src.get("poly_strict"),
        duree=round(duree, 3), tempo=round(tempo, 1), tonalite=av.tonalite(notes), nb_voix=k,
        nb_voix_auto=k_auto, polyphonie_max=int(poly.max()),
        polyphonie_moy=round(float(poly[poly > 0].mean()), 2),
        battements=[round(float(b), 3) for b in beats],
        voix=voix, accords=av.accords(notes, beats, duree), melodie=melodie,
        notes=[dict(s=round(n["s"], 3), e=round(n["e"], 3), p=n["p"], v=n["v"], voix=n["voix"],
                    nom=av.nom_note(n["p"])) for n in notes],
    )
    (dossier / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    av.ecrire_midi(notes, k, tempo, dossier)
    try:
        av.ecrire_musicxml(dossier, k, tempo)
    except Exception as ex:
        print("MusicXML échoué :", ex)
    print(json.dumps({x: data[x] for x in ("duree", "tempo", "tonalite", "nb_voix", "nb_voix_auto",
                                           "polyphonie_max", "polyphonie_moy")}, ensure_ascii=False))
    for v in voix:
        print(v["nom"], v["bas"], "→", v["haut"], "centre", v["mediane"], v["nb"], "notes")
    print(len(notes), "notes,", len(data["accords"]), "accords,", len(melodie), "notes de mélodie")


if __name__ == "__main__":
    main()
