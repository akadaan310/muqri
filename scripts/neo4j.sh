#!/usr/bin/env bash
# The local Neo4j + GDS server for the knowledge graph (localhost only; see GRAPH.md).
#   scripts/neo4j.sh start|stop|status
set -euo pipefail
export JAVA_HOME="${JAVA_HOME:-$HOME/opt/jre21}"
exec "$HOME/opt/neo4j/bin/neo4j" "${1:-status}"
