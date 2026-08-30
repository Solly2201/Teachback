import { STATE_META, studentStateLabel } from '../services/api.js'

/* The learning-state chip.

   `audience` picks the wording, not the state. Faculty views keep the formal
   state names the model is defined on; student views describe the EVIDENCE
   instead ("Very little evidence yet" rather than "Not Trying"), because the
   system observes answers and self-reports and never the student's intent. */
export default function StateBadge({ label, size = 'md', audience = 'faculty' }) {
  if (!label) return <span className="text-xs text-charcoal-light">No data yet</span>
  const meta = STATE_META[label] || STATE_META['Unclear']
  const sizeCls = size === 'lg' ? 'text-base px-4 py-1.5' : 'text-xs px-2.5 py-1'
  const text = audience === 'student' ? studentStateLabel(label) : label
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-bold uppercase tracking-wide ${meta.color} ${sizeCls}`}>
      <span aria-hidden="true">{meta.icon}</span>
      {text}
    </span>
  )
}
