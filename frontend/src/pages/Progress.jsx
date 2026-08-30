import { useEffect, useState } from 'react'
import StateBadge from '../components/StateBadge.jsx'
import StateTimeline from '../components/StateTimeline.jsx'
import { api } from '../services/api.js'

export default function Progress({ user }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.progress(user.id).then(setData).catch((e) => setError(e.message))
  }, [user.id])

  if (error) return <div className="card p-6 text-red-700">Failed to load progress: {error}</div>
  if (!data) return <div className="text-charcoal-light p-6">Loading progress…</div>

  const timeline = data.timeline

  return (
    <div className="space-y-5">
      <div className="banner">Learning Progress</div>

      <div className="card">
        <div className="card-header">State trajectory across sessions</div>
        <div className="p-4">
          <StateTimeline timeline={timeline} height={240} />
        </div>
        <div className="px-4 pb-4 text-xs text-charcoal-light border-t border-zinc-100 pt-3">
          Each point is one TeachBack session. States are estimated by a Hidden Markov Model over the whole
          sequence, so a new session can also refine the interpretation of earlier ones. States are a snapshot of
          your current learning condition — not a permanent label.
        </div>
      </div>

      <div className="card">
        <div className="card-header">Why your state changed — session by session</div>
        <div className="p-4">
          <div className="space-y-0">
            {[...timeline].slice(-6).reverse().map((o, idx, arr) => (
              <div key={o.id} className="flex gap-4">
                {/* vertical connector */}
                <div className="flex flex-col items-center">
                  <span className="w-2.5 h-2.5 rounded-full bg-brand mt-1.5" />
                  {idx < arr.length - 1 && <span className="flex-1 w-px bg-zinc-200" />}
                </div>
                <div className={idx < arr.length - 1 ? 'pb-5 flex-1' : 'flex-1'}>
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span className="text-xs font-bold uppercase tracking-wide text-charcoal-light">
                      Session {timeline.length - idx}
                    </span>
                    <StateBadge label={o.state_label} audience="student" />
                    <span className="text-xs text-charcoal-light">
                      {o.topic_name || ''}{o.created_at ? ` · ${new Date(o.created_at).toLocaleDateString()}` : ''}
                    </span>
                  </div>
                  {o.evidence?.length > 0 && (
                    <ul className="mt-1.5 space-y-0.5">
                      {o.evidence.map((e, i) => (
                        <li key={i} className="text-sm text-charcoal-light flex items-start gap-1.5">
                          <span className="text-zinc-400 mt-0.5">•</span>{e}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="px-4 pb-4 text-xs text-charcoal-light border-t border-zinc-100 pt-3">
          The bullets are the evidence from each session (concept coverage, accuracy, effort, misconceptions) —
          the HMM turns that sequence of evidence into your learning state.
        </div>
      </div>

      {data.summaries?.length > 0 && (
        <div className="card">
          <div className="card-header">Your lecture takeaways</div>
          <div className="divide-y divide-zinc-100">
            {data.summaries.map((s, i) => (
              <div key={i} className="p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold text-charcoal">{s.topic_name || 'Session'}</div>
                  <span className="text-xs text-charcoal-light whitespace-nowrap">
                    {s.created_at ? new Date(s.created_at).toLocaleDateString() : ''}
                  </span>
                </div>
                <p className="text-sm text-charcoal-light mt-1 italic">“{s.text}”</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.completions?.length > 0 && (
        <div className="card">
          <div className="card-header">Completed activities</div>
          <div className="divide-y divide-zinc-100">
            {data.completions.map((c) => (
              <div key={c.id} className="p-3.5 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5">
                  <span className="text-emerald-600">✓</span>
                  <div>
                    <div className="text-sm font-medium text-charcoal">{c.title}</div>
                    <div className="text-xs text-charcoal-light">
                      {(c.kind || '').replace('_', ' ')}{c.topic_name ? ` · ${c.topic_name}` : ''}
                    </div>
                  </div>
                </div>
                <span className="text-xs text-charcoal-light whitespace-nowrap">
                  {c.created_at ? new Date(c.created_at).toLocaleDateString() : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">Session history</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-charcoal-light border-b border-zinc-200">
                <th className="px-4 py-2.5">#</th>
                <th className="px-4 py-2.5">Date</th>
                <th className="px-4 py-2.5">Topic</th>
                <th className="px-4 py-2.5">Learning state</th>
                <th className="px-4 py-2.5">Coverage</th>
                <th className="px-4 py-2.5">Correctness</th>
                <th className="px-4 py-2.5">Effort</th>
                <th className="px-4 py-2.5">Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {[...timeline].reverse().map((o, idx) => (
                <tr key={o.id} className="hover:bg-zinc-50">
                  <td className="px-4 py-2.5 text-charcoal-light tabular-nums">{timeline.length - idx}</td>
                  <td className="px-4 py-2.5 text-charcoal-light whitespace-nowrap">
                    {o.created_at ? new Date(o.created_at).toLocaleDateString() : '—'}
                  </td>
                  <td className="px-4 py-2.5 font-medium text-charcoal">{o.topic_name || '—'}</td>
                  <td className="px-4 py-2.5"><StateBadge label={o.state_label} audience="student" /></td>
                  <td className="px-4 py-2.5 tabular-nums">{Math.round(o.features[0] * 100)}%</td>
                  <td className="px-4 py-2.5 tabular-nums">{Math.round(o.features[1] * 100)}%</td>
                  <td className="px-4 py-2.5 tabular-nums">{Math.round(o.features[4] * 100)}%</td>
                  <td className="px-4 py-2.5 text-xs text-charcoal-light">{o.source === 'live' ? 'TeachBack' : 'seeded'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
