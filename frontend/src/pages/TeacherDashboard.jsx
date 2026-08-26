import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import StateBadge from '../components/StateBadge.jsx'
import { TeacherContextBar, useTeacherContext } from '../components/TeacherContext.jsx'
import { api, STATE_META } from '../services/api.js'

export default function TeacherDashboard() {
  const context = useTeacherContext()
  const [data, setData] = useState(null)
  const [evaluation, setEvaluation] = useState(null)
  const [error, setError] = useState(null)

  // every aggregate is scoped by the selected subject on the backend —
  // switching subject refetches, so no cross-subject data can appear here
  useEffect(() => {
    if (!context.subject) return
    setData(null)
    api.teacherOverview(context.subject.id).then(setData).catch((e) => setError(e.message))
  }, [context.subject?.id]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    api.evaluation().then(setEvaluation).catch(() => {})
  }, [])

  if (error) return <div className="card p-6 text-red-700">Failed to load overview: {error}</div>
  if (!context.teachers) return <div className="text-charcoal-light p-6">Loading class overview…</div>
  if (!data)
    return (
      <div className="space-y-5">
        <div className="banner">Class Overview — {context.subject?.name || ''}</div>
        <TeacherContextBar context={context} />
        <div className="text-charcoal-light p-6">Loading class overview…</div>
      </div>
    )

  const maxCount = Math.max(...data.distribution.map((d) => d.count), 1)
  const maxMiscon = Math.max(...data.common_misconceptions.map((m) => m.count), 1)
  const topicFeedback = data.topic_feedback || []

  return (
    <div className="space-y-5">
      <div className="banner">Class Overview — {context.subject?.name || ''}</div>
      <TeacherContextBar context={context} />
      <div className="card p-4 flex items-center justify-between gap-4">
        <p className="text-sm text-charcoal-light">
          After a lecture, create a TeachBack from your material — the system drafts the concepts,
          you review them, students explain what they understood.
        </p>
        <Link to="/lectures" className="btn-primary whitespace-nowrap">+ Create Lecture TeachBack</Link>
      </div>

      <div className="grid sm:grid-cols-3 gap-4">
        <div className="card p-5 text-center">
          <div className="text-3xl font-black text-brand tabular-nums">{data.student_count}</div>
          <div className="text-xs uppercase tracking-wide text-charcoal-light mt-1">Students in this subject</div>
        </div>
        <div className="card p-5 text-center">
          <div className="text-3xl font-black text-brand tabular-nums">{data.topic_stats.length}</div>
          <div className="text-xs uppercase tracking-wide text-charcoal-light mt-1">Active topics</div>
        </div>
        <div className="card p-5 text-center">
          <div className="text-3xl font-black text-brand tabular-nums">{data.live_session_count}</div>
          <div className="text-xs uppercase tracking-wide text-charcoal-light mt-1">Live TeachBack sessions</div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        {/* state distribution */}
        <div className="card">
          <div className="card-header">Current class state distribution</div>
          <div className="p-5 space-y-3">
            {data.distribution.map((d) => (
              <div key={d.key} className="flex items-center gap-3">
                <div className="w-44 text-xs text-charcoal shrink-0 flex items-center gap-1.5">
                  <span style={{ color: STATE_META[d.state]?.dot }}>{STATE_META[d.state]?.icon}</span>
                  {d.state}
                </div>
                <div className="flex-1 h-4 bg-zinc-100 rounded overflow-hidden">
                  <div
                    className="h-full bg-brand rounded-r"
                    style={{ width: `${(d.count / maxCount) * 100}%` }}
                  />
                </div>
                <div className="w-16 text-right text-xs font-semibold text-charcoal tabular-nums">
                  {d.percent}% <span className="text-charcoal-light font-normal">({d.count})</span>
                </div>
              </div>
            ))}
            <p className="text-xs text-charcoal-light pt-2 border-t border-zinc-100">
              Based on each student&apos;s most recent HMM-estimated state.
            </p>
          </div>
        </div>

        {/* misconceptions */}
        <div className="card">
          <div className="card-header">Common misconceptions</div>
          <div className="p-5 space-y-3">
            {data.common_misconceptions.length === 0 && (
              <p className="text-sm text-charcoal-light">No misconceptions recorded yet.</p>
            )}
            {data.common_misconceptions.map((m) => (
              <div key={m.name} className="flex items-center gap-3">
                <div className="flex-1 text-sm text-charcoal">{m.name}</div>
                <div className="w-28 h-2.5 bg-zinc-100 rounded-full overflow-hidden shrink-0">
                  <div className="h-full bg-charcoal-light rounded-full" style={{ width: `${(m.count / maxMiscon) * 100}%` }} />
                </div>
                <div className="w-8 text-right text-xs font-semibold text-charcoal tabular-nums">{m.count}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        {/* declining students */}
        <div className="card">
          <div className="card-header">Students needing attention</div>
          <div className="divide-y divide-zinc-100">
            {data.declining_students.length === 0 && (
              <div className="p-4 text-sm text-charcoal-light">No student shows a deteriorating state right now.</div>
            )}
            {data.declining_students.map((s) => (
              <div key={s.id} className="p-4 flex items-center justify-between gap-3">
                <div className="font-semibold text-sm text-charcoal">{s.name}</div>
                <div className="text-xs text-charcoal-light flex items-center gap-1.5">
                  <span>{s.from_state}</span>
                  <span className="text-brand font-bold">→</span>
                  <StateBadge label={s.to_state} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* recent interactions */}
        <div className="card">
          <div className="card-header">Recent interactions</div>
          <div className="divide-y divide-zinc-100 max-h-80 overflow-y-auto">
            {data.recent_interactions.map((o) => (
              <div key={o.id} className="p-3.5 flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-charcoal">{o.student_name}</div>
                  <div className="text-xs text-charcoal-light">
                    {o.topic_name} · {o.created_at ? new Date(o.created_at).toLocaleString() : ''}
                  </div>
                </div>
                <StateBadge label={o.state_label} />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* lecture feedback */}
      {topicFeedback.length > 0 && (
        <div className="card">
          <div className="card-header">Lecture feedback from students</div>
          <div className="p-4 grid md:grid-cols-2 gap-4">
            {topicFeedback.map((f) => {
              const paceTotal = f.pace.reduce((n, p) => n + p.count, 0) || 1
              return (
                <div key={f.id} className="border border-zinc-200 rounded-md p-4">
                  <div className="font-semibold text-sm text-charcoal">{f.name}</div>
                  <div className="text-xs text-charcoal-light mt-0.5">{f.responses} feedback response{f.responses === 1 ? '' : 's'}</div>
                  <div className="flex gap-6 mt-3 text-sm">
                    {f.avg_confidence != null && (
                      <div><span className="font-bold text-charcoal tabular-nums">{f.avg_confidence}</span><span className="text-charcoal-light">/10 confidence</span></div>
                    )}
                    {f.avg_difficulty != null && (
                      <div><span className="font-bold text-charcoal tabular-nums">{f.avg_difficulty}</span><span className="text-charcoal-light">/10 difficulty</span></div>
                    )}
                  </div>
                  {f.pace.length > 0 && (
                    <div className="mt-3 space-y-1">
                      {f.pace.map((p) => (
                        <div key={p.label} className="flex items-center gap-2 text-xs">
                          <span className="w-24 text-charcoal-light">{p.label}</span>
                          <div className="flex-1 h-2.5 bg-zinc-100 rounded overflow-hidden">
                            <div className="h-full bg-brand rounded-r" style={{ width: `${(p.count / paceTotal) * 100}%` }} />
                          </div>
                          <span className="w-10 text-right tabular-nums text-charcoal">{Math.round((p.count / paceTotal) * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {f.common_requests.length > 0 && (
                    <div className="mt-3 text-xs text-charcoal-light">
                      <span className="font-semibold text-charcoal">Common requests:</span>{' '}
                      {f.common_requests.map((c) => c.label).join(' · ')}
                    </div>
                  )}
                  {f.recent_comments.length > 0 && (
                    <ul className="mt-2 space-y-0.5">
                      {f.recent_comments.map((c, i) => (
                        <li key={i} className="text-xs text-charcoal-light italic">“{c}”</li>
                      ))}
                    </ul>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* knowledge-check performance: MCQ and TeachBack shown side by side,
          deliberately never merged into one "mastery" number */}
      {(data.knowledge_checks || []).length > 0 && (
        <div className="card">
          <div className="card-header">Knowledge check performance</div>
          <div className="p-4 grid md:grid-cols-2 gap-4">
            {data.knowledge_checks.map((k) => (
              <div key={k.id} className="border border-zinc-200 rounded-md p-4">
                <div className="flex items-baseline justify-between gap-3">
                  <div className="font-semibold text-sm text-charcoal">{k.name}</div>
                  <div className="text-sm"><span className="font-bold text-charcoal tabular-nums">{k.avg_percent}%</span>
                    <span className="text-charcoal-light text-xs"> avg · {k.attempts} attempt{k.attempts === 1 ? '' : 's'}</span>
                  </div>
                </div>
                <table className="w-full text-xs mt-3">
                  <thead>
                    <tr className="text-left uppercase tracking-wide text-charcoal-light border-b border-zinc-100">
                      <th className="py-1.5">Concept</th>
                      <th className="py-1.5 text-right">Knowledge check</th>
                      <th className="py-1.5 text-right">TeachBack demonstrated</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-50">
                    {k.concepts.map((c) => (
                      <tr key={c.name}>
                        <td className="py-1.5 text-charcoal">{c.name}</td>
                        <td className="py-1.5 text-right tabular-nums text-charcoal">{c.mcq_percent}%</td>
                        <td className="py-1.5 text-right tabular-nums text-charcoal">
                          {c.teachback_percent != null ? `${c.teachback_percent}%` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-[11px] text-charcoal-light mt-2">
                  Two separate signals: what students could recognise (check) vs what they could explain (TeachBack).
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* topic stats */}
      <div className="card">
        <div className="card-header">Topic-level statistics</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-charcoal-light border-b border-zinc-200">
                <th className="px-4 py-2.5">Topic</th>
                <th className="px-4 py-2.5">Sessions</th>
                <th className="px-4 py-2.5">Avg concept coverage</th>
                <th className="px-4 py-2.5">Avg misconception score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {data.topic_stats.map((t) => (
                <tr key={t.id} className="hover:bg-zinc-50">
                  <td className="px-4 py-2.5 font-medium text-charcoal">{t.name}</td>
                  <td className="px-4 py-2.5 tabular-nums">{t.sessions}</td>
                  <td className="px-4 py-2.5 tabular-nums">{Math.round(t.avg_concept_coverage * 100)}%</td>
                  <td className="px-4 py-2.5 tabular-nums">{Math.round(t.avg_misconception_score * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* model evaluation footnote */}
      {evaluation?.available && (
        <div className="card p-4 text-xs text-charcoal-light leading-relaxed">
          <strong className="text-charcoal">Model evaluation (on held-out synthetic data & labelled responses):</strong>{' '}
          HMM state accuracy {Math.round(evaluation.hmm.state_accuracy * 100)}% ({evaluation.hmm.n_test_sessions} test sessions,{' '}
          adjacent-state {Math.round(evaluation.hmm.adjacent_state_accuracy * 100)}%) · concept detection F1{' '}
          {evaluation.nlp.concept_detection.f1} · misconception detection F1 {evaluation.nlp.misconception_detection.f1}{' '}
          (precision {evaluation.nlp.misconception_detection.precision}, recall {evaluation.nlp.misconception_detection.recall}).
        </div>
      )}
    </div>
  )
}
