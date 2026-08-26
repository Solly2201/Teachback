import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import StateBadge from '../components/StateBadge.jsx'
import StateTimeline from '../components/StateTimeline.jsx'
import { api, STATE_DESCRIPTIONS } from '../services/api.js'

export default function StudentDashboard({ user }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.student(user.id).then(setData).catch((e) => setError(e.message))
  }, [user.id])

  if (error) return <div className="card p-6 text-red-700">Failed to load dashboard: {error}</div>
  if (!data) return <div className="text-charcoal-light p-6">Loading dashboard…</div>

  const rec = data.recommendation

  return (
    <div className="space-y-5">
      <div className="banner">Student Dashboard</div>

      <div className="grid lg:grid-cols-3 gap-5">
        {/* current state */}
        <div className="card lg:col-span-2">
          <div className="card-header">Current learning state</div>
          <div className="p-5 flex flex-col sm:flex-row sm:items-center gap-4">
            <StateBadge label={data.current_state_label} size="lg" />
            <p className="text-sm text-charcoal-light flex-1">
              {STATE_DESCRIPTIONS[data.current_state_label] ||
                'Complete your first TeachBack session so the system can estimate your learning state.'}
            </p>
            <Link to="/teachback" className="btn-primary whitespace-nowrap text-center">
              Start TeachBack
            </Link>
          </div>
          <div className="px-5 pb-4 text-xs text-charcoal-light border-t border-zinc-100 pt-3">
            Estimated by a Hidden Markov Model from your last {data.session_count} sessions — it changes as you do.
          </div>
        </div>

        {/* recommended activity */}
        <div className="card">
          <div className="card-header">Recommended next activity</div>
          <div className="p-5">
            {rec ? (
              <>
                <div className="text-xs font-bold uppercase tracking-wide text-brand mb-1">{rec.activity.kind.replace('_', ' ')}</div>
                <div className="font-semibold text-charcoal">{rec.activity.title}</div>
                <p className="text-sm text-charcoal-light mt-1.5">{rec.activity.description}</p>
                {rec.why && (
                  <div className="mt-3 pt-3 border-t border-zinc-100">
                    <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-1">Why</div>
                    <p className="text-xs text-charcoal">{rec.why}</p>
                  </div>
                )}
                {rec.topic_name && <div className="text-xs text-charcoal-light mt-3">Topic: {rec.topic_name}</div>}
                {rec.notes?.map((n, i) => (
                  <div key={i} className="mt-3 text-xs bg-amber-50 border border-amber-200 text-amber-800 rounded-md p-2">{n}</div>
                ))}
              </>
            ) : (
              <p className="text-sm text-charcoal-light">Complete a session to get a recommendation.</p>
            )}
          </div>
        </div>
      </div>

      {/* timeline */}
      <div className="card">
        <div className="card-header">Learning-state timeline</div>
        <div className="p-4">
          <StateTimeline timeline={data.timeline} />
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        {/* recent topics */}
        <div className="card">
          <div className="card-header">Recent topics</div>
          <div className="divide-y divide-zinc-100">
            {data.recent_topics.length === 0 && <div className="p-4 text-sm text-charcoal-light">No topics yet.</div>}
            {data.recent_topics.map((t) => (
              <div key={t.id} className="p-4 flex items-center justify-between">
                <div className="font-semibold text-sm text-charcoal">{t.name}</div>
                <StateBadge label={t.last_state} />
              </div>
            ))}
          </div>
        </div>

        {/* recent activity */}
        <div className="card">
          <div className="card-header">Recent activity</div>
          <div className="divide-y divide-zinc-100 max-h-72 overflow-y-auto">
            {[...data.timeline].reverse().slice(0, 6).map((o) => (
              <div key={o.id} className="p-3.5 flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-charcoal">{o.topic_name || 'Session'}</div>
                  <div className="text-xs text-charcoal-light">
                    {o.created_at ? new Date(o.created_at).toLocaleString() : ''} · {o.source === 'live' ? 'TeachBack session' : 'past session'}
                  </div>
                </div>
                <StateBadge label={o.state_label} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
