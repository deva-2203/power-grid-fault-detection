import { useMemo, useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis,
  ReferenceArea, ReferenceLine, ResponsiveContainer
} from 'recharts'

const ZONE_AREAS = [
  { y1: -1.5, y2:  1.5, fill: 'rgba(46,213,115,0.06)'  },
  { y1:  1.5, y2:  2.5, fill: 'rgba(255,165,2,0.06)'   },
  { y1: -2.5, y2: -1.5, fill: 'rgba(255,165,2,0.06)'   },
  { y1:  2.5, y2:  5.0, fill: 'rgba(255,71,87,0.08)'   },
  { y1: -5.0, y2: -2.5, fill: 'rgba(255,71,87,0.08)'   },
]

function MiniChart({ data, dataKey, color, label }) {
  return (
    <div className="w-full">
      <div className="flex justify-between items-center px-1 mb-0.5">
        <span className="text-[9px] font-bold uppercase tracking-widest" style={{ color }}>
          {label}
        </span>
        {data.length > 0 && (
          <span className="text-[9px] font-mono text-slate-500">
            {data[data.length - 1][dataKey]?.toFixed(2)}σ
          </span>
        )}
      </div>
      <div style={{ height: 64 }}>
        <ResponsiveContainer width="100%" height={64}>
          <LineChart data={data} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
            {ZONE_AREAS.map((z, i) => (
              <ReferenceArea key={i} y1={z.y1} y2={z.y2} fill={z.fill} />
            ))}
            <ReferenceLine y={0}    stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
            <ReferenceLine y={ 1.5} stroke="rgba(255,165,2,0.35)"   strokeDasharray="3 3" strokeWidth={0.8} />
            <ReferenceLine y={-1.5} stroke="rgba(255,165,2,0.35)"   strokeDasharray="3 3" strokeWidth={0.8} />
            <ReferenceLine y={ 2.5} stroke="rgba(255,71,87,0.45)"   strokeDasharray="3 3" strokeWidth={0.8} />
            <ReferenceLine y={-2.5} stroke="rgba(255,71,87,0.45)"   strokeDasharray="3 3" strokeWidth={0.8} />
            <XAxis dataKey="step" hide />
            <YAxis
              domain={[-4.5, 4.5]}
              ticks={[-2.5, 0, 2.5]}
              tickFormatter={v => v === 0 ? 'Norm' : (v > 0 ? 'Over' : 'Under')}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 8, fill: '#475569' }}
              width={34}
            />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function NodeTile({ node, history }) {
  const [warning, setWarning] = useState(null)
  const dismissTimer = useRef(null)

  const data = useMemo(() => {
    return history.map(h => ({
      step: h._step,
      tau: h.sensors[node.tau] ?? 0,
      p:   h.sensors[node.p]   ?? 0,
      g:   h.sensors[node.g]   ?? 0,
    }))
  }, [history, node])

  const status = useMemo(() => {
    if (!data.length) return { label: 'Normal', color: '#2ed573', bg: '46,213,115' }
    const last = data[data.length - 1]
    const maxZ = Math.max(Math.abs(last.tau), Math.abs(last.p), Math.abs(last.g))
    if (maxZ < 1.5) return { label: 'Normal',   color: '#2ed573', bg: '46,213,115' }
    if (maxZ < 2.5) return { label: 'Elevated',  color: '#ffa502', bg: '255,165,2'  }
    return              { label: 'Critical',  color: '#ff4757', bg: '255,71,87'  }
  }, [data])

  // Fire a 10-second sticky warning whenever any sensor breaches a threshold
  useEffect(() => {
    if (!data.length) return
    const last = data[data.length - 1]
    const powerLabel = node.id === 0 ? 'Power Output' : 'Power Demand'

    const checks = [
      { key: 'tau', z: last.tau, label: 'Reaction Time' },
      { key: 'p',   z: last.p,   label: powerLabel },
      { key: 'g',   z: last.g,   label: 'Grid Cooperation' },
    ]

    // Find worst breaching metric
    let worst = null
    for (const c of checks) {
      const absZ = Math.abs(c.z)
      if (absZ >= 2.5) {
        if (!worst || absZ > Math.abs(worst.z)) worst = { ...c, tier: 'Critical', color: '#ff4757', bg: '255,71,87' }
      } else if (absZ >= 1.5) {
        if (!worst || (Math.abs(c.z) > Math.abs(worst.z) && worst.tier !== 'Critical'))
          worst = { ...c, tier: 'Elevated', color: '#ffa502', bg: '255,165,2' }
      }
    }

    if (worst) {
      const direction = worst.z > 0 ? 'too high' : 'too low'
      setTimeout(() => {
        setWarning({
          tier:    worst.tier,
          color:   worst.color,
          bg:      worst.bg,
          message: `${worst.label} is ${direction} (${worst.z.toFixed(2)}σ)`,
        })
      }, 0)

      // Clear existing timer, restart 10s countdown
      if (dismissTimer.current) clearTimeout(dismissTimer.current)
      dismissTimer.current = setTimeout(() => setWarning(null), 10000)
    }

    return () => {} // no cleanup needed; timer ref handles it
  }, [data, node])

  const powerLabel = node.id === 0 ? 'Power Output' : 'Power Demand'

  return (
    <div
      className="flex flex-col bg-slate-800/30 border border-slate-700/50 rounded-xl overflow-hidden backdrop-blur-md"
      style={{ borderTop: `3px solid ${node.accent}` }}
    >
      {/* Tile header */}
      <div className="flex justify-between items-center px-3 py-2">
        <div>
          <div className="text-[9px] font-bold tracking-widest text-slate-500 uppercase">
            NODE {node.id}
          </div>
          <div className="text-sm font-bold text-slate-100 flex items-center gap-1.5 mt-0.5">
            <span>{node.icon}</span>
            <span>{node.role}</span>
          </div>
        </div>
        <div
          className="text-[10px] font-bold px-2 py-0.5 rounded-full border"
          style={{
            color: status.color,
            backgroundColor: `rgba(${status.bg},0.15)`,
            borderColor: `rgba(${status.bg},0.35)`,
          }}
        >
          {status.label}
        </div>
      </div>

      {/* Sticky 10-second warning bar */}
      <AnimatePresence>
        {warning && (
          <motion.div
            key="warning"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div
              className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-bold"
              style={{
                backgroundColor: `rgba(${warning.bg}, 0.15)`,
                borderTop:    `1px solid rgba(${warning.bg}, 0.25)`,
                borderBottom: `1px solid rgba(${warning.bg}, 0.25)`,
                color: warning.color,
              }}
            >
              <span className="text-base leading-none">
                {warning.tier === 'Critical' ? '🔴' : '⚠️'}
              </span>
              <span>{warning.tier}: {warning.message}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Three separate mini-charts — unaffected by warning */}
      <div className="flex flex-col gap-1 px-2 pb-2 divide-y divide-slate-700/30">
        <MiniChart data={data} dataKey="tau" color="#3d8ef8" label="Reaction Time" />
        <div className="pt-1">
          <MiniChart data={data} dataKey="p" color="#ffa502" label={powerLabel} />
        </div>
        <div className="pt-1">
          <MiniChart data={data} dataKey="g" color="#2ed573" label="Grid Cooperation" />
        </div>
      </div>
    </div>
  )
}
