const CLUSTER_STYLES = {
  'normal operation': {
    color: '#2ed573',
    bg: 'rgba(46,213,115,0.13)',
    border: 'rgba(46,213,115,0.32)',
  },
  'pre-fault stress': {
    color: '#ffa502',
    bg: 'rgba(255,165,2,0.13)',
    border: 'rgba(255,165,2,0.32)',
  },
  'active fault': {
    color: '#ff4757',
    bg: 'rgba(255,71,87,0.13)',
    border: 'rgba(255,71,87,0.32)',
  },
}

export function clusterStyle(cluster) {
  return CLUSTER_STYLES[cluster] || {
    color: '#94a3b8',
    bg: 'rgba(148,163,184,0.12)',
    border: 'rgba(148,163,184,0.28)',
  }
}
