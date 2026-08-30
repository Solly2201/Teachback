import { useState } from 'react'
import { STATE_ORDER, studentStateLabel } from '../services/api.js'

/**
 * Learning-state trajectory: session index (x) vs ordinal state (y).
 * Single series -> one brand-red step line with markers, no legend needed.
 */
export default function StateTimeline({ timeline, height = 220, audience = 'student' }) {
  const [hover, setHover] = useState(null)
  const points = timeline.filter((o) => o.state_index !== null && o.state_index !== undefined)
  if (points.length === 0) {
    return <div className="text-sm text-charcoal-light p-6 text-center">No sessions yet — complete a TeachBack session to see your trajectory.</div>
  }

  const padL = 150
  const padR = 24
  const padT = 16
  const padB = 44
  const width = 720
  const innerW = width - padL - padR
  const innerH = height - padT - padB
  const n = points.length
  const x = (i) => padL + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW)
  const y = (s) => padT + innerH - (s / 4) * innerH

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(p.state_index)}`).join(' ')

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[560px]" role="img" aria-label="Learning state over sessions">
        {/* recessive horizontal gridlines + state labels */}
        {STATE_ORDER.map((name, s) => (
          <g key={name}>
            <line x1={padL} x2={width - padR} y1={y(s)} y2={y(s)} stroke="#E4E4E7" strokeWidth="1" />
            <text x={padL - 8} y={y(s) + 4} textAnchor="end" fontSize="11" fill="#666666">
              {audience === 'student' ? studentStateLabel(name) : name}
            </text>
          </g>
        ))}
        {/* line */}
        <path d={path} fill="none" stroke="#A5231B" strokeWidth="2" strokeLinejoin="round" />
        {/* markers with white ring */}
        {points.map((p, i) => (
          <g key={p.id ?? i}>
            <circle
              cx={x(i)} cy={y(p.state_index)} r={hover === i ? 7 : 5}
              fill="#A5231B" stroke="#FFFFFF" strokeWidth="2"
            />
            {/* generous invisible hit target */}
            <rect
              x={x(i) - 14} y={padT} width="28" height={innerH} fill="transparent"
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
            />
            <text x={x(i)} y={height - 26} textAnchor="middle" fontSize="11" fill="#666666">
              {i + 1}
            </text>
          </g>
        ))}
        <text x={padL + innerW / 2} y={height - 8} textAnchor="middle" fontSize="10" fill="#A1A1AA">Session</text>
        {/* tooltip */}
        {hover !== null && (
          <g pointerEvents="none">
            {(() => {
              const p = points[hover]
              const tx = Math.min(Math.max(x(hover), padL + 90), width - padR - 90)
              const ty = Math.max(y(p.state_index) - 14, 46)
              return (
                <g>
                  <rect x={tx - 88} y={ty - 34} width="176" height="34" rx="5" fill="#3F3F46" />
                  <text x={tx} y={ty - 20} textAnchor="middle" fontSize="11" fill="#FFFFFF" fontWeight="bold">
                    {audience === 'student' ? studentStateLabel(p.state_label) : p.state_label}
                  </text>
                  <text x={tx} y={ty - 7} textAnchor="middle" fontSize="10" fill="#D4D4D8">
                    {p.topic_name || 'Session'} · {p.created_at ? new Date(p.created_at).toLocaleDateString() : `#${hover + 1}`}
                  </text>
                </g>
              )
            })()}
          </g>
        )}
      </svg>
    </div>
  )
}
