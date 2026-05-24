"""Single source of truth for the cloud runner's deployment-specific
identifiers. Every value defaults to the motley-team-475011 setup we
ship with, and every value is overridable via an environment variable
so other deployments don't have to fork the code.

Override env vars (all optional):

  BIRD_INTERACT_CLOUD_PROJECT      GCP project ID
  BIRD_INTERACT_CLOUD_REGION       GCP region (used for bucket + AR + zone family)
  BIRD_INTERACT_CLOUD_ZONE         GCE zone for the cluster
  BIRD_INTERACT_CLOUD_BUCKET       GCS bucket for runs/<id>/
  BIRD_INTERACT_CLOUD_AR_REPO      Artifact Registry repo name
  BIRD_INTERACT_CLOUD_WORKER_SA    Worker service account email (defaults to
                                    `<AR_REPO>@<PROJECT>.iam.gserviceaccount.com`)
  BIRD_INTERACT_CLOUD_IMAGE_NAME   Container image name inside AR repo

NOTE: every module-level constant below is resolved at IMPORT time, and the
derived ones read other constants — `ZONE` defaults to `f"{REGION}-a"` and
`WORKER_SA` to `f"{AR_REPO}@{PROJECT}.iam.gserviceaccount.com"`. So setting
only `BIRD_INTERACT_CLOUD_REGION` (without `_ZONE`) won't change `ZONE` unless
the module is re-imported; likewise `_PROJECT`/`_AR_REPO` vs `WORKER_SA`. To
pick up an override at runtime either set the derived var explicitly, set the
env BEFORE first import, or `importlib.reload` this module. `image_uri_prefix()`
is the exception — it recomputes from the env on every call.
"""

from __future__ import annotations

import os


PROJECT: str = os.environ.get("BIRD_INTERACT_CLOUD_PROJECT", "motley-team-475011")
REGION: str = os.environ.get("BIRD_INTERACT_CLOUD_REGION", "us-central1")
ZONE: str = os.environ.get("BIRD_INTERACT_CLOUD_ZONE", f"{REGION}-a")

BUCKET_NAME: str = os.environ.get(
    "BIRD_INTERACT_CLOUD_BUCKET", "motley-team-birdbench"
)
AR_REPO: str = os.environ.get(
    "BIRD_INTERACT_CLOUD_AR_REPO", "bird-interact-runner"
)
IMAGE_NAME: str = os.environ.get(
    "BIRD_INTERACT_CLOUD_IMAGE_NAME", "runner"
)

WORKER_SA: str = os.environ.get(
    "BIRD_INTERACT_CLOUD_WORKER_SA",
    f"{AR_REPO}@{PROJECT}.iam.gserviceaccount.com",
)


def image_uri_prefix() -> str:
    """Full Artifact Registry URI prefix for the runner image.
    Recomputed on each call so env-var overrides made in a test fixture
    are visible without a re-import."""
    project = os.environ.get("BIRD_INTERACT_CLOUD_PROJECT", PROJECT)
    region = os.environ.get("BIRD_INTERACT_CLOUD_REGION", REGION)
    repo = os.environ.get("BIRD_INTERACT_CLOUD_AR_REPO", AR_REPO)
    image = os.environ.get("BIRD_INTERACT_CLOUD_IMAGE_NAME", IMAGE_NAME)
    return f"{region}-docker.pkg.dev/{project}/{repo}/{image}"
