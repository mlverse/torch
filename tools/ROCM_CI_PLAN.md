# Plan de migration -- `make_rocm_kit.sh` vers GitHub Actions

Statut : proposition
Portee : `tools/make_rocm_kit.sh`, `tools/kit-lib.sh`, assets ROCm, CI
Date : 2026-11-10

## Principe directeur

Ne pas reecrire les ~1 500 lignes de `kit-lib.sh` en YAML. Le script reste le
moteur (teste, reversible via `--unpatch`) ; GitHub Actions l'orchestre via une
**action composite + un workflow reutilisable**. La migration consiste a mettre
le flux existant en CI, pas a porter la logique dans des steps YAML.

## 1. Inventaire des dependances a embarquer

| Element | Role en CI |
|---|---|
| `tools/make_rocm_kit.sh` | point d'entree (CLI, staging, build, copie) |
| `tools/kit-lib.sh` | staging, patchs, audit, doctor, quarantaine |
| `tools/hip-shim.cmake` | injection CMake `hip::*` |
| `tools/hip-compat/**` | shims `cuda* -> hip*` |
| `tools/cuda-hip-rocm.R` | patch R (`R/cuda.R`) |
| `tools/test-kit.sh` | 21 tests unitaires (sans GPU) |
| `tools/test_torch_rocm.R` | smoke test runtime (GPU requis) |
| `tools/README-rocm.md` / `tools/make_rocm_kit.CHANGELOG.md` | documentation |

## 2. Decoupage cible

```
.github/
  actions/build-rocm-lantern/action.yml   # composite : deps + appel du script
  workflows/rocm-kit.yaml                 # declencheurs + matrice + jobs
tools/make_rocm_kit.sh                    # inchange + option --ci
```

### Action composite `build-rocm-lantern`

- **inputs**
  - `rocm-version` (defaut : `6.4`)
  - `torch-version` (defaut : `2.8.0`)
  - `rocm-components` (vide = defaut leger du package, `all` = tout)
  - `libtorch-url` (vide = URL derivee de `rocm-version`/`torch-version`)
  - `cudatoolkit-ref` (defaut : `rocm6.4`)
  - `lantern-src`
  - `force`
  - `upload-artifact`
- **outputs**
  - `kit-dir`
  - `liblantern-path`
  - `artifact-name`
- **steps**
  1. installer les outils : `cmake rsync patchelf unzip` + binutils
  2. installer le paquet R `rocm6.4` (branche ROCm de `{cudatoolkit}`) ;
     `ROCM_PATH` = racine du package installe
  3. cacher/telecharger l'archive libtorch ROCm (`download.pytorch.org`),
     aucun venv Python
  4. lancer `tools/make_rocm_kit.sh --from-zip <archive> --ci`
  5. auditer (`readelf`, `patchelf`, dependances du `.so`)
  6. publier les artefacts

### Workflow `rocm-kit.yaml`

Declencheurs :

- `workflow_call`
- `workflow_dispatch` avec inputs (`rocm-version`, `torch-version`,
  `rocm-components`, `libtorch-url`, `cudatoolkit-ref`, `force`)
- `pull_request` filtre sur `tools/**`, `.github/actions/build-rocm-lantern/**`
- `schedule` nightly

## 3. Jobs

1. **`unit`** -- `ubuntu-latest`, sans ROCm
   - `bash -n` sur les scripts
   - `shellcheck` si disponible
   - `tools/test-kit.sh`
   - Objectif : proteger les helpers, feedback rapide.

2. **`build`** -- `ubuntu-22.04`
   - installer ROCm 6.4 **headers/dev** (aucun GPU requis pour compiler)
   - recuperer/cacher le wheel `torch 2.8.0+rocm6.4`
   - executer le kit
   - publier `liblantern.so`, `libcaffe2_nvrtc.so`, logs d'audit et tarball
     `torch-rocm-kit`.

3. **`smoke-load`** -- CPU
   - installer le paquet R depuis l'artefact
   - verifier `TORCH_URL` / `LANTERN_URL`
   - charger `liblantern` (dlopen) sans lancer de noyau GPU.

4. **`gpu-smoke`** -- runner self-hosted AMD, nightly/manual
   - executer `tools/test_torch_rocm.R`
   - appliquer `HSA_OVERRIDE_GFX_VERSION` si necessaire.

5. **`publish`** -- optionnel, hors fork
   - reutiliser `upload-s3` avec le nommage de `lantern.yaml` /
     `packaging.yaml`.

## 4. Prerequis bloquants a corriger avant/avec la migration

D'apres `tools/REQUESTED_CHANGES.md`, la CI ROCm actuelle est **inerte et
trompeuse** :

- `lantern.yaml` : entree `rocm6.4` sans `include` -> `matrix.rocm` vide,
  `Install ROCm` ne s'execute jamais.
- `Jimver/cuda-toolkit` n'a pas d'input `rocm` ; `matrix.config.cmake` n'est
  jamais defini -> build CPU etiquete ROCm.
- ancien `.github/actions/install-rocm/action.yml` (supprime) : input
  `rocm_version` ignore, `$ROCM_URL` jamais defini, depot `xenial`, cle GPG en
  HTTP sans `signed-by`.
- `ensure_hip_compat` peut melanger headers 7.x et wheel 6.4 (a rendre fatal ou
  explicitement autorise via `--allow-hip-header-mismatch`).
- Patchs R/C++ non gates sur la presence de `libtorch_hip.so`.
- `find | head` sous `pipefail` (SIGPIPE 141).
- `.gitignore` incomplet (`torch-rocm-kit/`, `*.orig-make-rocm-kit`,
  `*.kit-applied`).
- `test-kit.sh` peut etre faussement vert si `kit-lib.sh` ne se source pas.

## 5. Adaptations du script pour la CI

- Ajouter `--ci` : sortie sans ANSI, pas de `cp -v`, chemins deterministes,
  logs archivables.
- En CI, **ne pas supprimer** les artefacts en cas d'echec : archiver les logs
  et deplacements de quarantaine avant `if: always()`.
- Ne PAS utiliser de venv Python. Le toolkit ROCm vient du paquet R
  `rocm6.4` (branche ROCm de `{cudatoolkit}`, composants extraits de
  `repo.radeon.com/rocm/apt/6.4`), et libtorch vient de l'archive officielle
  `download.pytorch.org/libtorch/rocm6.4/`. Le kit recoit `--from-zip` +
  `ROCM_PATH` au lieu de `--from-venv`.
- Lire la version attendue depuis `version.py` au lieu du `2.8.0+rocm6.4` code
  en dur dans le message de staging.
- Ajouter un echec explicite sur le mismatch SDK/wheel au lieu d'un simple
  `warn`.

## 6. Ordre d'execution propose

| Phase | Contenu | Sortie |
|---|---|---|
| 0 | Corriger les bloqueurs CI + rendre le script CI-ready (`--ci`, gates ROCm, `.gitignore`) | base saine |
| 1 | Action composite + workflow minimal `build` sur dispatch | artefact `liblantern.so` reproductible |
| 2 | Job `unit` sur PR + path filters + cache wheel/SDK | garde-fou automatique |
| 3 | Runner self-hosted AMD + `gpu-smoke` nightly | validation runtime reelle |
| 4 | Integration S3 / `lantern.yaml` / `packaging.yaml` + doc | release ROCm |

## 7. Risques principaux

- **Taille/temps** : wheel ROCm ~2--5 Go, build 30--50 min -> cache, timeout
  60--90 min, liberer l'espace disque du runner.
- **Pas de GPU sur runners GitHub heberges** : le build passe, le test runtime
  doit etre self-hosted.
- **Mismatch SDK** : installer explicitement les headers 6.4 et echouer si
  `torch.version.hip` differe.
- **Securite apt** : HTTPS + `gpg --dearmor` + `signed-by`, versions epinglees.
- **Artefacts de fork** : ne jamais publier sur S3 depuis une PR externe.

## 8. Decisions ouvertes

1. Conserver le script comme source de verite (recommande) ou porter la logique
   dans des actions ?
2. Heberger un runner AMD self-hosted pour `gpu-smoke`, ou se limiter au build
   et au smoke-load CPU ?
3. Publier les artefacts sur S3 depuis la CI, ou seulement `upload-artifact`
   en attendant l'integration release ?

## 9. Etat d'implementation (v25)

Phases 0 et 1 ecrites ; le build CI n'a pas encore tourne faute de runner
ROCm. Le script reste la source de verite.

- [x] Phase 0 -- bloqueurs CI
  - [x] `install-rocm` supprime : le toolkit vient du paquet R `rocm6.4`
        (branche ROCm de `{cudatoolkit}`), qui extrait les `.deb` versionnes
        de `repo.radeon.com/rocm/apt/6.4` sans root (aucun apt, aucun venv).
  - [x] `lantern.yaml` : ligne `rocm6.4` inerte et input `rocm:` invalide
        supprimes (ROCm releve desormais de `rocm-kit.yaml`) ; `cmake`
        defini dans chaque entree de matrice.
  - [x] `--ci` : pas d'ANSI, pas de `cp -v`, `KIT_KEEP_ARTIFACTS=1` +
        `NO_COLOR=1` ; artefacts en echec deplaces vers
        `build-lantern/quarantine/` pour upload `if: always()`.
  - [x] Mismatch SDK/wheel HIP fatal (opt-out
        `--allow-hip-header-mismatch`).
  - [x] `apply_cuda_r_block` gate sur `libtorch_hip.so`.
  - [x] `find | head` protege sous `pipefail`.
  - [x] `.gitignore` : `torch-rocm-kit/`, `*.orig-make-rocm-kit`,
        `*.kit-applied`.
  - [x] `test-kit.sh` : echec si `kit-lib.sh` ne se source pas, si un helper
        manque, ou si zero test a tourne.
  - [x] Version lue depuis `version.py` (plus de `2.8.0+rocm6.4` en dur).
- [x] Phase 1 -- action composite + workflow minimal
  - [x] `.github/actions/build-rocm-lantern/action.yml` (deps, paquet R
        `rocm6.4`, archive libtorch + cache, `--from-zip --ci`, audit
        ELF/RPATH, artefacts + logs).
  - [x] `.github/workflows/rocm-kit.yaml` (`unit`, `build`, placeholders).
- [ ] Phase 2 -- job `unit` sur PR (deja present) + cache wheel/SDK + smoke
      load CPU.
- [ ] Phase 3 -- runner self-hosted AMD + `gpu-smoke` nightly.
- [ ] Phase 4 -- S3 / release.

### Decisions prises

1. Le script reste la source de verite (recommandation de la section 8).
2. `gpu-smoke` reste desactive tant qu'aucun runner AMD self-hosted n'est
   provisionne ; seuls `unit` et `build` tournent.
3. Publication via `upload-artifact` uniquement, aucune publication S3.
4. La ligne ROCm de `lantern.yaml` est retiree plutot que "reparee" : ce
   workflow telecharge un libtorch CPU/CUDA et ne peut pas produire un
   artefact ROCm. `rocm-kit.yaml` est le seul chemin ROCm.

### Points ouverts

- Epinglage exact des paquets apt ROCm : le depot versionne est utilise, mais
  les meta-paquets ne sont pas epingles a `=6.4.*` (les noms de versions
  varient). La verification `.info/version` echoue si apt resout une autre
  generation.
- `hipblaslt-dev`/`hipfort-dev` doivent exister dans le depot 6.4 ; sinon
  retirer ces paquets de la liste.

## 10. Decision : toolkit ROCm via `{cudatoolkit}` (pas de venv Python)

La CI n'utilisera pas de venv Python. Deux briques R remplacent l'ancien
`pip install torch` :

1. **`rocm6.4`** -- nouvelle variante ROCm de `{cudatoolkit}` (mlverse).
   Elle telecharge et extrait les paquets `.deb` ROCm 6.4 depuis
   `https://repo.radeon.com/rocm/apt/6.4` (composants : hip-runtime-amd,
   hsa-rocr, comgr, rocblas, hipblas/hipblaslt, hipfft, hipsparse,
   hipsolver, rocrand/hiprand, rccl, miopen, + les `-dev` pour les headers).
   API : `lib_path()`, `include_path(component)`, `bin_path(component)`,
   `rocm_path(component)`.
2. **libtorch ROCm** -- archive officielle
   `https://download.pytorch.org/libtorch/rocm6.4/libtorch-shared-with-deps-<version>%2Brocm6.4.zip`
   (3,7 Go pour 2.8.0), deja supportee par `--from-zip`. Aucun Python.
   Le kit recoit `--from-zip <archive>` + `ROCM_PATH=<lib R>/rocm6.4` ; pas de
   mode `--from-r-package` ni de venv.

Travail prealable fait dans le depot `{cudatoolkit}` (branche `rocm6.4`) :
scaffold `rocm6.4/`, `shared-rocm/`, `resolve-rocm.sh`, `sync-rocm.sh`,
`ROCM.md`, job CI `test-rocm`. Layout valide (teste avec
`core,hipblas-common,hip-dev,hsa-dev`) :

```
<lib R>/rocm6.4/
  include/                 # miroir de /opt/rocm/include (hip/, hsa/, rocblas/, ...)
  lib/                     # toutes les .so (RPATH $ORIGIN...)
  .info/version            # 6.4.0-47
  rocm/<component>/
    bin/ share/ lib/cmake/ .info/version
```

Donc `ROCM_PATH=<lib R>/rocm6.4` fonctionne directement avec `ensure_hip_compat`
(`$ROCM_PATH/include`, `$ROCM_PATH/.info/version`), sans Python. API :
`include_path()` (racine partagee), `lib_path()`, `rocm_path(component)`,
`bin_path(component)`. Scripts uniquement POSIX sh + awk.

Decisions :
- composants : `.deb` de `repo.radeon.com` (pas de wheel torch) ;
- defaut leger : core, runtime, hsa, hip-dev, device-libs, comgr,
  rocblas/hipblas/hipfft/hipsparse/hipsolver/rocrand/hiprand + leurs `-dev` ;
  `miopen`, `rccl`, `hipblaslt` et Thrust/CUB/rocPRIM en opt-in ;
  `ROCM_COMPONENTS=all` pour tout installer ;
- libtorch : archive `download.pytorch.org` via `--from-zip` ;
- cache CI : `ROCM_DEB_CACHE` + cache de l'archive libtorch.

Reste a valider :
- compatibilite avec `load_cudatoolkit_libs()` de `{torch}` (nom du package et
  detection ROCm vs CUDA) ;
- taille CI (rocblas ~3,6 Go installe, zip libtorch ~3,7 Go).
