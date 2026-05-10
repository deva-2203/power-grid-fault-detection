import { useState, useEffect, useRef, useMemo } from 'react'
import { Activity, Play, Square } from 'lucide-react'
import NodeTile from './components/NodeTile'

const API_URL = 'http://localhost:8000/api'
const SPEED_MS = 1000
const MAX_HIST = 60

const NODES = [
  { id: 0, role: "Power Generator", tau: "tau1", p: "p1", g: "g1", icon: "⚡", accent: "#3d8ef8" },
  { id: 1, role: "Industrial Load", tau: "tau2", p: "p2", g: "g2", icon: "🏭", accent: "#ffa502" },
  { id: 2, role: "Residential Load", tau: "tau3", p: "p3", g: "g3", icon: "🏠", accent: "#a78bfa" },
  { id: 3, role: "Commercial Load", tau: "tau4", p: "p4", g: "g4", icon: "🏢", accent: "#34d399" },
]

const RAW_FEATS = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]

export default function App() {
  const [feed, setFeed] = useState([])
  const [stats, setStats] = useState(null)
  const [step, setStep] = useState(0)
  const [running, setRunning] = useState(false)

  // Fetch initial data
  useEffect(() => {
    async function loadData() {
      try {
        const [statsRes, feedRes] = await Promise.all([
          fetch(`${API_URL}/stats`),
          fetch(`${API_URL}/feed/dataset`)
        ])
        const st = await statsRes.json()
        const fd = await feedRes.json()
        setStats(st)
        setFeed(fd)
        setLoading(false)
      } catch (err) {
        console.error("Failed to load data:", err)
      }
    }
    loadData()
  }, [])

  // Derived history — rolling window of last MAX_HIST readings
  const history = useMemo(() => {
    if (!feed.length || !stats) return []
    const start = Math.max(0, step - MAX_HIST + 1)
    return feed.slice(start, step + 1).map((current, idx) => {
      const raw = current.raw_input || {}
      const sensors = {}
      if (current.raw_input) {
        RAW_FEATS.forEach(f => {
          const mean = stats.mean[f] || 0
          const std = Math.max(stats.std[f] || 1e-6, 1e-6)
          const val = raw[f] || mean
          sensors[f] = (val - mean) / std
        })
      } else {
        RAW_FEATS.forEach(f => sensors[f] = 0)
      }
      return {
        _step: start + idx,
        prob: current.ensemble_prob,
        tier: current.risk_tier,
        sensors
      }
    })
  }, [feed, step, stats])

  // Playback loop
  useEffect(() => {
    let interval;
    if (running && step < feed.length - 1 && stats) {
      interval = setInterval(() => {
        setStep(s => s + 1)
      }, SPEED_MS)
    } else if (step >= feed.length - 1) {
      setRunning(false)
    }
    return () => clearInterval(interval)
  }, [running, step, feed, stats])

  if (!feed.length) {
    return <div className="flex h-screen items-center justify-center text-slate-400">Loading Grid Data...</div>
  }

  const current = feed[step] || {}

  return (
    <div className="min-h-screen p-6 max-w-7xl mx-auto space-y-6 pb-20">
      
      {/* Header */}
      <header className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-3 tracking-tight">
            <Activity className="text-blue-500" />
            Grid Fault Monitor
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Real-time multi-agent power grid stability prediction
          </p>
        </div>
        
        <div className="flex items-center gap-3 bg-slate-800/50 p-2 border border-slate-700/50 rounded-xl backdrop-blur-md">
            <div className="text-xs text-slate-400 font-mono px-2">
              RDG {step} / {feed.length}
            </div>
            <button 
              onClick={() => { setStep(0); setRunning(false); }}
              className="px-3 py-2 rounded-lg font-semibold text-xs text-slate-400 border border-slate-600 hover:bg-slate-700 transition-all"
            >
              ↺ Reset
            </button>
            <button 
              onClick={() => setRunning(!running)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm transition-all w-28 justify-center ${
                running 
                  ? 'bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30' 
                  : 'bg-blue-500 text-white shadow-[0_0_15px_rgba(59,130,246,0.3)] hover:bg-blue-400'
              }`}
            >
              {running ? <><Square size={16} /> Stop</> : <><Play size={16} /> Start</>}
            </button>
          </div>
      </header>

      <h2 className="text-lg font-semibold text-slate-200 mt-8 mb-4">Node Health — Live Sensor Feed</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {NODES.map(node => (
          <NodeTile key={node.id} node={node} history={history} />
        ))}
      </div>

    </div>
  )
}
