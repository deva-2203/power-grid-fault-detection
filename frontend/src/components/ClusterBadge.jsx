import { clusterStyle } from '../clusterStyles'

export default function ClusterBadge({ cluster, className = '' }) {
  const style = clusterStyle(cluster)

  return (
    <span
      className={`inline-flex items-center whitespace-nowrap rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${className}`}
      style={{ color: style.color, backgroundColor: style.bg, borderColor: style.border }}
    >
      {cluster || 'unknown'}
    </span>
  )
}
