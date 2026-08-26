import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { api } from '../services/api.js'

/* The recommended-activity page. Opened from a recommendation card with the
   activity passed via router state ({ activity, why }). Stored activities
   (with an id) are re-fetched from the API as the source of truth; generic
   fallback activities render from the passed payload. */
export default function Activity({ user }) {
  const location = useLocation()
  const seed = location.state?.activity || null
  const why = location.state?.why || null
  const [activity, setActivity] = useState(seed && !seed.id ? seed : null)
  const [answer, setAnswer] = useState('')
  const [done, setDone] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (seed?.id) api.activity(seed.id).then(setActivity).catch((e) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const complete = async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await api.completeActivity({
        student_id: user.id,
        activity_id: activity.id ?? null,
        topic_id: activity.topic_id ?? seed?.topic_id ?? null,
        title: activity.title,
        kind: activity.kind,
        answer: answer.trim(),
      })
      setDone(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  if (!seed) {
    return (
      <div className="space-y-5">
        <div className="banner">Activity</div>
        <div className="card p-6 text-sm text-charcoal-light">
          No activity selected. Open an activity from a recommendation on your dashboard or session summary.
        </div>
        <Link to="/" className="btn-secondary inline-block">Return to Dashboard</Link>
      </div>
    )
  }

  if (done) {
    return (
      <div className="space-y-5">
        <div className="banner">Activity</div>
        <div className="card p-8 text-center">
          <div className="text-3xl text-emerald-600 mb-2">✓</div>
          <div className="text-lg font-bold text-charcoal">Activity completed</div>
          <p className="text-sm text-charcoal-light mt-1.5">You completed {done.title}.</p>
          <div className="flex justify-center gap-3 mt-6">
            <Link to="/progress" className="btn-primary">Back to Progress</Link>
            <Link to="/" className="btn-secondary">Return to Dashboard</Link>
          </div>
        </div>
      </div>
    )
  }

  if (error && !activity) return <div className="card p-6 text-red-700">Failed to load activity: {error}</div>
  if (!activity) return <div className="text-charcoal-light p-6">Loading activity…</div>

  const topicName = activity.topic_name || seed?.topic_name

  return (
    <div className="space-y-5">
      <div className="banner">Activity — {activity.title}</div>
      {error && <div className="card p-4 text-sm text-red-700 bg-red-50 border-red-200">{error}</div>}

      <div className="card">
        <div className="card-header">
          <span>{activity.title}</span>
          {topicName && <span className="normal-case font-normal text-white/80">{topicName}</span>}
        </div>
        <div className="p-5 space-y-4">
          <div className="text-xs font-bold uppercase tracking-wide text-brand">
            {(activity.kind || 'practice').replace('_', ' ')}
          </div>
          {activity.description && <p className="text-sm text-charcoal-light">{activity.description}</p>}

          {activity.content && (
            <div className="border border-zinc-200 bg-zinc-50 rounded-md p-4 text-sm text-charcoal leading-relaxed whitespace-pre-line">
              {activity.content}
            </div>
          )}

          {why && (
            <div className="pt-3 border-t border-zinc-100">
              <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-1">Why this was recommended</div>
              <p className="text-xs text-charcoal">{why}</p>
            </div>
          )}

          <div className="pt-3 border-t border-zinc-100">
            <label className="label">Your task</label>
            <p className="text-sm font-semibold text-charcoal mb-2">
              {activity.question || 'When you have done the activity, note your answer or main takeaway below.'}
            </p>
            <textarea
              className="input min-h-[72px] resize-y"
              placeholder="One or two sentences is enough…"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              disabled={busy}
            />
          </div>

          <div className="flex items-center justify-between">
            <Link to="/" className="btn-secondary">Cancel</Link>
            <button onClick={complete} disabled={busy || !answer.trim()} className="btn-primary">
              {busy ? 'Saving…' : 'Submit & complete'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
