/** Thin horizontal meter for a 0-1 feature or percentage, with direct label. */
export default function Meter({ label, value, display, color = '#A5231B' }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div className="flex items-center gap-3 py-1">
      <div className="w-44 text-xs text-charcoal-light shrink-0">{label}</div>
      <div className="flex-1 h-2.5 bg-zinc-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <div className="w-12 text-right text-xs font-semibold text-charcoal tabular-nums">{display ?? `${pct}%`}</div>
    </div>
  )
}
