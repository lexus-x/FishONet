# Family names for each species — for Leah

**What this does:** finds the biological *family* for every fish species in the
dataset. Example: *Lepidotrigla spiloptera* -> **Triglidae**.

## To run
Open a terminal **in this folder** and type:

    bash run.sh

That's it. Takes ~2 minutes.

## What you get (both are ALREADY here, pre-filled)
- `family_by_species.csv` — open in Excel: two columns, *species* and its *family*.
- `family_map.json` — the same thing for code (species -> family).

Running `run.sh` just re-generates them fresh from GBIF.

## Good to know
- Source: **GBIF** (public global species database) — only taxonomy *names*,
  which is allowed by the competition rules.
- Covers **100%** of the 17,393 classes.
- **97%** of the unseen species are in a family we also have in training — that's
  why family is a useful signal. Next step (ask Sai): use family as a prediction prior.
