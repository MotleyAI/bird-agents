"""DEV-1822: render the mounted Cube config dir (spike-validated).

One multitenant `cube.js`: the JWT `securityContext.db` selects both the model
dir (per-tenant FileRepository) and the target Postgres database (driverFactory).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CUBE_JS_TEMPLATE = """const { FileRepository } = require('@cubejs-backend/server-core');

module.exports = {
  contextToAppId: ({ securityContext }) => `CUBE_APP_${securityContext.db}`,
  contextToOrchestratorId: ({ securityContext }) => `CUBE_ORC_${securityContext.db}`,
  repositoryFactory: ({ securityContext }) =>
    new FileRepository(`model/${securityContext.db}`),
  driverFactory: ({ securityContext }) => ({
    type: 'postgres',
    host: process.env.CUBEJS_DB_HOST,
    port: Number(process.env.CUBEJS_DB_PORT || 5432),
    database: securityContext.db,
    user: process.env.CUBEJS_DB_USER,
    password: process.env.CUBEJS_DB_PASS,
  }),
  scheduledRefreshContexts: async () => [],
};
"""


def render_conf(root: Path) -> Path:
    """Write `<root>/conf/cube.js` (+ empty `model/`); return the conf dir."""
    conf_dir = Path(root) / "conf"
    (conf_dir / "model").mkdir(parents=True, exist_ok=True)
    (conf_dir / "cube.js").write_text(CUBE_JS_TEMPLATE)
    return conf_dir


def conf_content_hash(root: Path) -> str:
    """Hash of the on-disk `cube.js` (falls back to the template)."""
    f = Path(root) / "conf" / "cube.js"
    content = f.read_text() if f.exists() else CUBE_JS_TEMPLATE
    return hashlib.sha256(content.encode()).hexdigest()
