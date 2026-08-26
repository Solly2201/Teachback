import { STATE_META } from '../services/api.js'

export default function StateBadge({ label, size = 'md' }) {
  if (!label) return <span className="text-xs text-charcoal-light">No data yet</span>
  const meta = STATE_META[label] || STATE_META['Unclear']
  const sizeCls = size === 'lg' ? 'text-base px-4 py-1.5' : 'text-xs px-2.5 py-1'
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-bold uppercase tracking-wide ${meta.color} ${sizeCls}`}>
      <span aria-hidden="true">{meta.icon}</span>
      {label}
    </span>
  )
}
