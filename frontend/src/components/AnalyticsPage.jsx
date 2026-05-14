import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CircleDot,
  GitBranch,
  Gauge,
  Loader2,
  Network,
  ShieldCheck,
  Zap,
} from 'lucide-react'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  Radar,
  RadarChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { clusterStyle } from '../clusterStyles'
import ClusterBadge from './ClusterBadge'
import { API_URL } from '../config'

const RAW_FEATS = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']

const TIER_COLORS = {
  Green: '#2ed573',
  Amber: '#ffa502',
  Red: '#ff4757',
}

const CLUSTER_ORDER = ['normal operation', 'pre-fault stress', 'active fault']

const FEATURE_GROUPS = [
  { key: 'tau', label: 'Reaction Time', color: '#3d8ef8' },
  { key: 'p', label: 'Power Balance', color: '#ffa502' },
  { key: 'g', label: 'Cooperation', color: '#2ed573' },
]

function Panel({ title, icon: Icon, children, className = '' }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-slate-800/30 border border-slate-700/50 rounded-xl p-4 backdrop-blur-md ${className}`}
    >
      <div className="flex items-center gap-2 mb-4">
        <Icon size={16} className="text-blue-400" />
        <h2 className="text-sm font-bold text-slate-200">{title}</h2>
      </div>
      {children}
    </motion.section>
  )
}

function StatCard({ label, value, sub, icon: Icon, color = '#f8fafc' }) {
  return (
    <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500">{label}</div>
          <div className="text-2xl font-bold mt-1" style={{ color }}>{value}</div>
        </div>
        <div className="w-9 h-9 rounded-lg border border-slate-700/60 bg-slate-900/50 flex items-center justify-center">
          <Icon size={17} style={{ color }} />
        </div>
      </div>
      {sub && <div className="text-xs text-slate-400 mt-2">{sub}</div>}
    </div>
  )
}

function tooltipStyle() {
  return {
    backgroundColor: '#0f172a',
    border: '1px solid rgba(100,116,139,0.45)',
    borderRadius: 10,
    color: '#e2e8f0',
  }
}

function pct(value) {
  return `${Math.round((value || 0) * 100)}%`
}

export default function AnalyticsPage() {
  const [feed, setFeed] = useState([])
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function loadAnalytics() {
      try {
        const [feedRes, statsRes] = await Promise.all([
          fetch(`${API_URL}/feed/dataset`),
          fetch(`${API_URL}/stats`),
        ])
        if (!feedRes.ok || !statsRes.ok) throw new Error('Backend analytics data is unavailable.')
        const [feedData, statsData] = await Promise.all([feedRes.json(), statsRes.json()])
        setFeed(feedData)
        setStats(statsData)
      } catch (err) {
        setError(err.message)
      }
    }
    loadAnalytics()
  }, [])

  const analytics = useMemo(() => {
    if (!feed.length || !stats) return null

    const tierCounts = { Green: 0, Amber: 0, Red: 0 }
    const clusterCounts = Object.fromEntries(CLUSTER_ORDER.map(cluster => [cluster, 0]))
    const featureStress = Object.fromEntries(RAW_FEATS.map(f => [f, { total: 0, max: 0 }]))
    const groupStress = Object.fromEntries(FEATURE_GROUPS.map(g => [g.key, { total: 0, count: 0 }]))
    let faultCount = 0
    let trueFaults = 0
    let correct = 0
    let maxProb = 0
    let maxRow = 0
    let probTotal = 0
    let rfTotal = 0
    let xgbTotal = 0
    let anomalyTotal = 0

    const trend = feed.map((row, index) => {
      const prob = row.ensemble_prob || 0
      const rf = row.RF_prob || 0
      const xgb = row.XGBoost_prob || 0
      const anomaly = row.IF_anomaly_score || 0
      const label = row.ensemble_label === 1 ? 1 : 0
      const truth = row._true_label === 1 ? 1 : 0

      tierCounts[row.risk_tier] = (tierCounts[row.risk_tier] || 0) + 1
      clusterCounts[row.kmeans_cluster] = (clusterCounts[row.kmeans_cluster] || 0) + 1
      faultCount += label
      trueFaults += truth
      correct += label === truth ? 1 : 0
      probTotal += prob
      rfTotal += rf
      xgbTotal += xgb
      anomalyTotal += anomaly
      if (prob > maxProb) {
        maxProb = prob
        maxRow = index + 1
      }

      RAW_FEATS.forEach(feature => {
        const mean = stats.mean?.[feature] || 0
        const std = Math.max(stats.std?.[feature] || 1e-6, 1e-6)
        const value = row.raw_input?.[feature] ?? mean
        const z = Math.abs((value - mean) / std)
        featureStress[feature].total += z
        featureStress[feature].max = Math.max(featureStress[feature].max, z)
        const group = feature.replace(/\d/g, '')
        groupStress[group].total += z
        groupStress[group].count += 1
      })

      return {
        row: index + 1,
        probability: Math.round(prob * 100),
        rf: Math.round(rf * 100),
        xgb: Math.round(xgb * 100),
        anomaly,
        tier: row.risk_tier,
      }
    })

    const sampleEvery = Math.max(1, Math.ceil(trend.length / 90))
    const sampledTrend = trend.filter((_, index) => index % sampleEvery === 0 || index === trend.length - 1)
    const tierData = Object.entries(tierCounts).map(([name, value]) => ({ name, value }))
    const clusterData = Object.entries(clusterCounts)
      .filter(([, value]) => value > 0)
      .map(([name, value]) => ({
        name,
        value,
        pct: value / feed.length,
        color: clusterStyle(name).color,
      }))
    const stressData = Object.entries(featureStress)
      .map(([feature, value]) => ({
        feature,
        mean: Number((value.total / feed.length).toFixed(2)),
        max: Number(value.max.toFixed(2)),
      }))
      .sort((a, b) => b.mean - a.mean)

    const radarData = FEATURE_GROUPS.map(group => ({
      group: group.label,
      stress: Number((groupStress[group.key].total / Math.max(groupStress[group.key].count, 1)).toFixed(2)),
      fullMark: 3,
    }))

    return {
      accuracy: correct / feed.length,
      faultRate: faultCount / feed.length,
      trueFaultRate: trueFaults / feed.length,
      meanProb: probTotal / feed.length,
      maxProb,
      maxRow,
      modelData: [
        { model: 'Random Forest', probability: Math.round((rfTotal / feed.length) * 100), color: '#38bdf8' },
        { model: 'XGBoost', probability: Math.round((xgbTotal / feed.length) * 100), color: '#a78bfa' },
        { model: 'Ensemble', probability: Math.round((probTotal / feed.length) * 100), color: '#2ed573' },
      ],
      anomalyMean: anomalyTotal / feed.length,
      sampledTrend,
      tierData,
      clusterData,
      stressData,
      radarData,
    }
  }, [feed, stats])

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-center gap-3 text-red-300">
        <AlertTriangle size={18} />
        <span className="text-sm">{error}</span>
      </div>
    )
  }

  if (!analytics) {
    return (
      <div className="flex h-64 flex-col items-center justify-center gap-3 text-slate-400">
        <Loader2 size={30} className="animate-spin text-blue-400" />
        <span className="text-sm">Loading analytics...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="Replay Rows" value={feed.length.toLocaleString()} sub="Precomputed test feed" icon={BarChart3} color="#38bdf8" />
        <StatCard label="Model Accuracy" value={pct(analytics.accuracy)} sub="Compared with test labels" icon={ShieldCheck} color="#2ed573" />
        <StatCard label="Fault Rate" value={pct(analytics.faultRate)} sub={`True label rate ${pct(analytics.trueFaultRate)}`} icon={Zap} color="#ff4757" />
        <StatCard label="Peak Risk" value={pct(analytics.maxProb)} sub={`Highest probability at row ${analytics.maxRow}`} icon={Gauge} color="#ffa502" />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <Panel title="Fault Probability Timeline" icon={BarChart3} className="xl:col-span-2">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={analytics.sampledTrend} margin={{ top: 10, right: 12, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="probFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.42} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.03} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
                <XAxis dataKey="row" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 100]} tickFormatter={value => `${value}%`} tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle()} formatter={(value, name) => [name === 'anomaly' ? Number(value).toFixed(3) : `${value}%`, name]} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                <Area type="monotone" dataKey="probability" name="Ensemble" stroke="#3b82f6" fill="url(#probFill)" strokeWidth={2.5} />
                <Line type="monotone" dataKey="rf" name="RF" stroke="#38bdf8" strokeWidth={1.4} dot={false} />
                <Line type="monotone" dataKey="xgb" name="XGBoost" stroke="#a78bfa" strokeWidth={1.4} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Risk Tier Mix" icon={GitBranch}>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={analytics.tierData} dataKey="value" nameKey="name" innerRadius={66} outerRadius={104} paddingAngle={3}>
                  {analytics.tierData.map(entry => (
                    <Cell key={entry.name} fill={TIER_COLORS[entry.name]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle()} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <Panel title="Operating Cluster Mix" icon={CircleDot} className="xl:col-span-2">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {analytics.clusterData.map(cluster => (
              <div key={cluster.name} className="bg-slate-900/35 border border-slate-700/50 rounded-lg p-4">
                <div className="flex items-center justify-between gap-3">
                  <ClusterBadge cluster={cluster.name} />
                  <span className="text-xl font-bold" style={{ color: cluster.color }}>
                    {Math.round(cluster.pct * 100)}%
                  </span>
                </div>
                <div className="mt-3 h-2 rounded-full bg-slate-700/70 overflow-hidden">
                  <div className="h-full rounded-full" style={{ width: `${cluster.pct * 100}%`, backgroundColor: cluster.color }} />
                </div>
                <div className="mt-2 text-xs text-slate-400">{cluster.value.toLocaleString()} replay rows</div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Cluster Meaning" icon={GitBranch}>
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between gap-3">
              <ClusterBadge cluster="normal operation" />
              <span className="text-xs text-slate-400">low-stress behavior</span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <ClusterBadge cluster="pre-fault stress" />
              <span className="text-xs text-slate-400">warning pattern</span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <ClusterBadge cluster="active fault" />
              <span className="text-xs text-slate-400">fault-like behavior</span>
            </div>
          </div>
        </Panel>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <Panel title="Top Sensor Stress" icon={Network} className="xl:col-span-2">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={analytics.stressData} margin={{ top: 6, right: 12, left: -20, bottom: 0 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
                <XAxis dataKey="feature" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle()} formatter={value => `${value}σ`} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="mean" name="Mean z-score" fill="#38bdf8" radius={[6, 6, 0, 0]} />
                <Bar dataKey="max" name="Peak z-score" fill="#ff4757" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Stress Profile" icon={BrainCircuit}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={analytics.radarData} outerRadius={92}>
                <PolarGrid stroke="rgba(148,163,184,0.2)" />
                <PolarAngleAxis dataKey="group" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                <PolarRadiusAxis angle={90} domain={[0, 3]} tick={{ fill: '#64748b', fontSize: 10 }} />
                <Radar dataKey="stress" stroke="#2ed573" fill="#2ed573" fillOpacity={0.28} />
                <Tooltip contentStyle={tooltipStyle()} formatter={value => `${value}σ`} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <Panel title="Model Probability Comparison" icon={BrainCircuit}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {analytics.modelData.map(model => (
            <div key={model.model} className="bg-slate-900/35 border border-slate-700/50 rounded-lg p-4">
              <div className="flex items-center justify-between text-sm font-semibold text-slate-200">
                <span>{model.model}</span>
                <span style={{ color: model.color }}>{model.probability}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-700/70 overflow-hidden mt-3">
                <div className="h-full rounded-full" style={{ width: `${model.probability}%`, backgroundColor: model.color }} />
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  )
}
