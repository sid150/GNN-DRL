/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WS_BASE_URL: string;
  readonly VITE_APP_TITLE: string;
  readonly VITE_ENABLE_WEBSOCKET: string;
  readonly VITE_ENABLE_AUTH: string;
  readonly VITE_ENABLE_EXPERIMENTS: string;
  readonly VITE_ENABLE_TRAINING: string;
  readonly VITE_POLLING_INTERVAL: string;
  readonly VITE_MAX_TOPOLOGY_NODES: string;
  readonly VITE_CHART_MAX_POINTS: string;
  readonly VITE_TABLE_PAGE_SIZE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
