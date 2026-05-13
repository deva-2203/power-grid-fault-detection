import { useState, useRef, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, FileSpreadsheet, Download, AlertTriangle, ShieldCheck, Flame, ChevronDown, ChevronUp, X, Loader2 } from 'lucide-react'

const API_URL = 'http://localhost:8000/api'
const RAW_FEATS = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]

const TIER_STYLES = {
  Green: { color: '#2ed573', bg: 'rgba(46,213,115,0.15)', border: 'rgba(46,213,115,0.35)', icon: ShieldCheck },
  Amber: { color: '#ffa502', bg: 'rgba(255,165,2,0.15)',  border: 'rgba(255,165,2,0.35)',  icon: AlertTriangle },
  Red:   { color: '#ff4757', bg: 'rgba(255,71,87,0.15)',  border: 'rgba(255,71,87,0.35)',  icon: Flame },
}

function ProbBar({ value }) {
  const pct = Math.round(value * 100)
  const color = value < 0.4 ? '#2ed573' : value <= 0.7 ? '#ffa502' : '#ff4757'
  return (
    <div className="flex items-center gap-2 min-w-[140px]">
      <div className="flex-1 h-2 rounded-full bg-slate-700/60 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        />
      </div>
      <span className="text-xs font-mono w-10 text-right" style={{ color }}>{pct}%</span>
    </div>
  )
}

function TierBadge({ tier }) {
  const s = TIER_STYLES[tier] || TIER_STYLES.Green
  const Icon = s.icon
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border"
      style={{ color: s.color, backgroundColor: s.bg, borderColor: s.border }}
    >
      <Icon size={10} /> {tier}
    </span>
  )
}

function SummaryCard({ label, value, sub, color }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 backdrop-blur-md"
    >
      <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">{label}</div>
      <div className="text-2xl font-bold" style={{ color: color || '#f8fafc' }}>{value}</div>
      {sub && <div className="text-xs text-slate-400 mt-0.5">{sub}</div>}
    </motion.div>
  )
}

export default function DatasetPredict() {
  const [results, setResults] = useState(null)
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [sortCol, setSortCol] = useState('row_index')
  const [sortAsc, setSortAsc] = useState(true)
  const [expandedRow, setExpandedRow] = useState(null)
  const fileRef = useRef(null)

  async function uploadFile(file) {
    if (!file) return
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file.')
      return
    }
    setLoading(true)
    setError(null)
    setResults(null)
    setSummary(null)
    setFileName(file.name)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API_URL}/predict/upload`, { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Server error ${res.status}`)
      }
      const data = await res.json()
      setResults(data.results)
      setSummary(data.summary)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) uploadFile(file)
  }

  function handleFileInput(e) {
    const file = e.target.files?.[0]
    if (file) uploadFile(file)
  }

  function downloadCSV() {
    if (!results) return
    const header = ['row', ...RAW_FEATS, 'ensemble_prob', 'risk_tier', 'label', 'kmeans_cluster', 'RF_prob', 'XGBoost_prob', 'IF_anomaly_score', 'alert_reason']
    const rows = results.map((r, i) => [
      i + 1,
      ...RAW_FEATS.map(f => r.raw_input?.[f] ?? ''),
      r.ensemble_prob?.toFixed(4),
      r.risk_tier,
      r.ensemble_label === 1 ? 'Fault' : 'Stable',
      r.kmeans_cluster,
      r.RF_prob?.toFixed(4),
      r.XGBoost_prob?.toFixed(4),
      r.IF_anomaly_score?.toFixed(4),
      (r.alert_reason || []).map(a => `${a.feature}(${a.z_score}σ)`).join('; ')
    ])
    const csv = [header.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `predictions_${fileName || 'output'}`
    a.click()
    URL.revokeObjectURL(url)
  }

  const sorted = useMemo(() => {
    if (!results) return []
    return [...results].sort((a, b) => {
      let va = a[sortCol], vb = b[sortCol]
      if (sortCol === 'risk_tier') { va = { Green: 0, Amber: 1, Red: 2 }[va] ?? 0; vb = { Green: 0, Amber: 1, Red: 2 }[vb] ?? 0 }
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va)
      return sortAsc ? va - vb : vb - va
    })
  }, [results, sortCol, sortAsc])

  function toggleSort(col) {
    if (sortCol === col) setSortAsc(!sortAsc)
    else { setSortCol(col); setSortAsc(false) }
  }

  const SortIcon = ({ col }) => {
    if (sortCol !== col) return <ChevronDown size={12} className="opacity-30" />
    return sortAsc ? <ChevronUp size={12} /> : <ChevronDown size={12} />
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      {!results && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer ${
            dragOver
              ? 'border-blue-400 bg-blue-500/10'
              : 'border-slate-600/50 bg-slate-800/20 hover:border-slate-500 hover:bg-slate-800/40'
          }`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={handleFileInput} />
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
              <Upload size={28} className="text-blue-400" />
            </div>
            <div>
              <p className="text-lg font-semibold text-slate-200">Drop your CSV here or click to browse</p>
              <p className="text-sm text-slate-400 mt-1">The model will predict fault probability for every row</p>
            </div>
            <div className="mt-4 bg-slate-800/60 border border-slate-700/50 rounded-xl p-4 text-left max-w-md w-full">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                <FileSpreadsheet size={14} /> Expected CSV Format
              </div>
              <code className="text-[11px] text-slate-300 leading-relaxed block font-mono">
                tau1,tau2,tau3,tau4,p1,p2,p3,p4,g1,g2,g3,g4{'\n'}
                2.959,0.928,0.311,8.279,3.763,−1.734,...{'\n'}
                4.127,1.582,6.390,1.851,2.419,−0.812,...
              </code>
            </div>
          </div>
        </motion.div>
      )}

      {/* Loading spinner */}
      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center justify-center py-20 gap-4"
        >
          <Loader2 size={40} className="text-blue-400 animate-spin" />
          <p className="text-slate-400 text-sm">Running inference on <span className="text-slate-200 font-semibold">{fileName}</span>...</p>
        </motion.div>
      )}

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-center gap-3"
          >
            <AlertTriangle size={18} className="text-red-400 flex-shrink-0" />
            <span className="text-sm text-red-300 flex-1">{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300"><X size={16} /></button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      {results && summary && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">

          {/* Action bar */}
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5">
                <FileSpreadsheet size={14} className="text-blue-400" />
                <span className="text-sm text-slate-300 font-medium">{fileName}</span>
                <span className="text-xs text-slate-500">({summary.total_rows} rows)</span>
              </div>
              <button
                onClick={() => { setResults(null); setSummary(null); setFileName(null); setExpandedRow(null) }}
                className="text-xs text-slate-400 hover:text-slate-200 border border-slate-600 rounded-lg px-3 py-1.5 hover:bg-slate-700 transition-all"
              >
                ↺ New Upload
              </button>
            </div>
            <button
              onClick={downloadCSV}
              className="flex items-center gap-2 bg-blue-500/15 border border-blue-500/30 text-blue-400 hover:bg-blue-500/25 rounded-lg px-4 py-1.5 text-sm font-semibold transition-all"
            >
              <Download size={14} /> Export CSV
            </button>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <SummaryCard label="Total Rows" value={summary.total_rows} />
            <SummaryCard label="Faults Detected" value={summary.fault_count} sub={`${summary.fault_pct}% of total`} color="#ff4757" />
            <SummaryCard label="Stable" value={summary.stable_count} sub={`${(100 - summary.fault_pct).toFixed(1)}% of total`} color="#2ed573" />
            <SummaryCard label="Mean Probability" value={`${Math.round(summary.mean_prob * 100)}%`} color={summary.mean_prob < 0.4 ? '#2ed573' : summary.mean_prob <= 0.7 ? '#ffa502' : '#ff4757'} />
            <SummaryCard
              label="Tier Breakdown"
              value={
                <div className="flex items-center gap-2 text-base">
                  <span style={{ color: '#2ed573' }}>{summary.tier_green}</span>
                  <span className="text-slate-600">/</span>
                  <span style={{ color: '#ffa502' }}>{summary.tier_amber}</span>
                  <span className="text-slate-600">/</span>
                  <span style={{ color: '#ff4757' }}>{summary.tier_red}</span>
                </div>
              }
              sub="Green / Amber / Red"
            />
          </div>

          {summary.rows_dropped > 0 && (
            <div className="text-xs text-amber-400/80 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2 flex items-center gap-2">
              <AlertTriangle size={14} /> {summary.rows_dropped} row{summary.rows_dropped > 1 ? 's' : ''} dropped due to missing values
            </div>
          )}

          {/* Results Table */}
          <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl overflow-hidden backdrop-blur-md">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    {[
                      { key: 'row_index', label: '#', width: 'w-12' },
                      { key: 'ensemble_prob', label: 'Fault Probability' },
                      { key: 'risk_tier', label: 'Tier' },
                      { key: 'ensemble_label', label: 'Prediction' },
                      { key: 'kmeans_cluster', label: 'Cluster' },
                      { key: 'RF_prob', label: 'RF Prob' },
                      { key: 'XGBoost_prob', label: 'XGB Prob' },
                      { key: 'IF_anomaly_score', label: 'Anomaly Score' },
                    ].map(col => (
                      <th
                        key={col.key}
                        onClick={() => toggleSort(col.key)}
                        className={`text-left px-3 py-2.5 text-[10px] font-bold uppercase tracking-widest text-slate-400 cursor-pointer hover:text-slate-200 transition-colors select-none ${col.width || ''}`}
                      >
                        <span className="inline-flex items-center gap-1">
                          {col.label} <SortIcon col={col.key} />
                        </span>
                      </th>
                    ))}
                    <th className="text-left px-3 py-2.5 text-[10px] font-bold uppercase tracking-widest text-slate-400">Alert</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((r) => (
                    <tr key={r.row_index} className="group">
                      {/* Main Row */}
                      <td className="px-3 py-2 text-xs text-slate-500 font-mono border-b border-slate-700/30">
                        <button
                          onClick={() => setExpandedRow(expandedRow === r.row_index ? null : r.row_index)}
                          className="hover:text-blue-400 transition-colors"
                          title="Show raw features"
                        >
                          {r.row_index + 1}
                        </button>
                      </td>
                      <td className="px-3 py-2 border-b border-slate-700/30"><ProbBar value={r.ensemble_prob} /></td>
                      <td className="px-3 py-2 border-b border-slate-700/30"><TierBadge tier={r.risk_tier} /></td>
                      <td className="px-3 py-2 border-b border-slate-700/30">
                        <span className={`text-xs font-semibold ${r.ensemble_label === 1 ? 'text-red-400' : 'text-emerald-400'}`}>
                          {r.ensemble_label === 1 ? 'Fault' : 'Stable'}
                        </span>
                      </td>
                      <td className="px-3 py-2 border-b border-slate-700/30 text-xs text-slate-300">{r.kmeans_cluster}</td>
                      <td className="px-3 py-2 border-b border-slate-700/30 text-xs text-slate-400 font-mono">{(r.RF_prob * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2 border-b border-slate-700/30 text-xs text-slate-400 font-mono">{(r.XGBoost_prob * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2 border-b border-slate-700/30 text-xs text-slate-400 font-mono">{r.IF_anomaly_score?.toFixed(3)}</td>
                      <td className="px-3 py-2 border-b border-slate-700/30">
                        <div className="flex flex-wrap gap-1">
                          {(r.alert_reason || []).slice(0, 2).map((a, j) => (
                            <span key={j} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-300 font-mono">
                              {a.feature} <span className={Math.abs(a.z_score) >= 2.5 ? 'text-red-400' : Math.abs(a.z_score) >= 1.5 ? 'text-amber-400' : 'text-slate-500'}>{a.z_score}σ</span>
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Expanded row detail (raw features) */}
          <AnimatePresence>
            {expandedRow !== null && results[expandedRow] && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 8 }}
                className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 backdrop-blur-md"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-bold uppercase tracking-widest text-slate-400">Row {expandedRow + 1} — Raw Features</span>
                  <button onClick={() => setExpandedRow(null)} className="text-slate-400 hover:text-slate-200"><X size={14} /></button>
                </div>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                  {RAW_FEATS.map(f => (
                    <div key={f} className="bg-slate-700/30 rounded-lg px-3 py-2">
                      <div className="text-[9px] font-bold uppercase tracking-widest text-slate-500">{f}</div>
                      <div className="text-sm font-mono text-slate-200">{results[expandedRow].raw_input?.[f]?.toFixed(4) ?? '—'}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </motion.div>
      )}
    </div>
  )
}
