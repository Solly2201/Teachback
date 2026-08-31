import { useEffect, useState } from 'react'
import StateBadge from '../components/StateBadge.jsx'
import { TeacherContextBar, useTeacherContext } from '../components/TeacherContext.jsx'
import { api } from '../services/api.js'

/* Student Evidence — the teacher-oversight screen.

   TeachBack assists the teacher; it does not replace them. So a teacher can
   walk Student -> session -> individual response -> the question asked -> what
   the system concluded, and disagree with any of it. Two things this page is
   built around:

   1. The student's exact words and the system's reading of them are never
      allowed to look like the same kind of statement. They get different
      labels, different backgrounds and different type.
   2. Nothing important is hover-only. Selecting a student opens a real panel.

   It is also where an evaluation is closed, because that is the moment the
   raw responses are destroyed — the teacher should be looking at them when
   they decide they are finished with them. */

const CONCEPT_MARK = { covered: '✓', partial: '◐', missing: '○' }
const CONCEPT_TONE = {
  covered: 'text-emerald-700',
  partial: 'text-amber-700',
  missing: 'text-charcoal-light',
}
const REL_TONE = {
  demonstrated: 'text-emerald-700',
  needs_clarification: 'text-amber-700',
  not_discussed: 'text-charcoal-light',
}

function Pill({ tone = 'zinc', children }) {
  const tones = {
    zinc: 'bg-zinc-100 text-charcoal-light border-zinc-200',
    amber: 'bg-amber-50 text-amber-800 border-amber-200',
    emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  }
  return (
    <span className={`text-[11px] font-bold uppercase tracking-wide px-2 py-0.5 rounded border ${tones[tone]}`}>
      {children}
    </span>
  )
}

/* Closing an evaluation destroys the raw responses for good. The dialog says
   that in those words — it is not an "archive" and must not read like one. */
function CloseEvaluationDialog({ preview, topicName, busy, onCancel, onConfirm }) {
  useEffect(() => {
    if (!preview) return undefined
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [preview, busy, onCancel])

  if (!preview) return null
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-charcoal/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Close evaluation confirmation"
      onClick={(e) => { if (e.target === e.currentTarget && !busy) onCancel() }}
    >
      <div className="card max-w-lg w-full p-6 space-y-4 max-h-[85vh] overflow-y-auto">
        <div className="text-lg font-bold text-charcoal">
          Close evaluation for {topicName}?
        </div>
        <p className="text-sm text-charcoal-light">
          This will stop new TeachBack sessions for this lecture.
          Before closing, review any student responses you want to inspect.
        </p>
        <div className="text-sm">
          <div className="font-semibold text-charcoal mb-1">When evaluation is closed:</div>
          <ul className="text-sm text-charcoal-light space-y-1 list-disc pl-5">
            <li>new TeachBack sessions cannot be started</li>
            <li>existing aggregate learning evidence is retained</li>
            <li><strong className="text-charcoal">individual student responses are permanently deleted</strong></li>
            <li><strong className="text-charcoal">raw free-text responses cannot be recovered</strong></li>
          </ul>
        </div>
        <div className="text-xs bg-red-50 border border-red-200 text-red-800 rounded p-3 space-y-1">
          <div className="font-bold uppercase tracking-wide">Permanently deleted</div>
          <ul className="space-y-0.5">
            {preview.removed.map((line) => <li key={line}>{line}</li>)}
          </ul>
        </div>
        <div className="text-xs bg-zinc-50 border border-zinc-200 text-charcoal-light rounded p-3 space-y-1">
          <div className="font-bold uppercase tracking-wide text-charcoal">Kept</div>
          <ul className="space-y-0.5">
            {preview.kept.map((line) => <li key={line}>{line}</li>)}
          </ul>
        </div>
        <p className="text-xs text-charcoal-light">
          This action does not delete the lecture, concepts, quiz results, or student progress.
        </p>
        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="btn-secondary" disabled={busy}>Cancel</button>
          <button
            onClick={onConfirm}
            disabled={busy}
            className="px-4 py-2 rounded-md font-semibold text-white bg-red-700 hover:bg-red-800 disabled:opacity-50"
          >
            {busy ? 'Closing…' : 'Close Evaluation'}
          </button>
        </div>
      </div>
    </div>
  )
}

/* One exchange. The two blocks below are deliberately not interchangeable:
   what the student said is quoted verbatim; what TeachBack made of it is
   labelled as an interpretation and can be checked against the lecture text. */
function ResponseCard({ response }) {
  return (
    <div className="border border-zinc-200 rounded-md overflow-hidden">
      <div className="bg-zinc-50 px-4 py-2 text-xs text-charcoal-light border-b border-zinc-200">
        Exchange {response.exchange_no}
        {response.contributed_to_coverage
          ? <span className="ml-2 text-emerald-700 font-semibold">contributed to coverage</span>
          : <span className="ml-2">no evidence added</span>}
      </div>
      <div className="p-4 space-y-3">
        <div>
          <div className="label">Question asked</div>
          <p className="text-sm text-charcoal">{response.question || '—'}</p>
        </div>
        <div>
          <div className="label">Student response</div>
          <blockquote className="text-sm text-charcoal border-l-4 border-brand bg-white pl-3 py-1.5 italic">
            {response.answer || '—'}
          </blockquote>
          <div className="text-[11px] text-charcoal-light mt-1">
            {response.word_count} word{response.word_count === 1 ? '' : 's'}, exactly as written
          </div>
        </div>
        <div className="bg-zinc-50 border border-zinc-200 rounded p-3 space-y-2">
          <div className="label !mb-0">System interpretation</div>
          <ul className="space-y-1.5">
            {response.concepts.map((c) => (
              <li key={c.name} className="text-sm">
                <span className={`font-semibold ${CONCEPT_TONE[c.status]}`}>
                  {CONCEPT_MARK[c.status]} {c.status_label}: {c.name}
                </span>
                <div className="text-xs text-charcoal-light mt-0.5">{c.why}</div>
                {c.lecture_reference && c.status !== 'missing' && (
                  <div className="text-xs text-charcoal-light mt-0.5">
                    <span className="font-semibold text-charcoal">From your lecture:</span>{' '}
                    {c.lecture_reference}
                  </div>
                )}
              </li>
            ))}
          </ul>
          {response.relationships.length > 0 && (
            <ul className="space-y-1.5 pt-2 border-t border-zinc-200">
              {response.relationships.map((r, i) => (
                <li key={i} className="text-sm">
                  <span className={`font-semibold ${REL_TONE[r.status]}`}>
                    {r.source} → {r.target}: {r.status_label}
                  </span>
                  <div className="text-xs text-charcoal-light mt-0.5">{r.why}</div>
                </li>
              ))}
            </ul>
          )}
          {response.misconception && (
            <div className="pt-2 border-t border-zinc-200 text-sm">
              <span className="font-semibold text-amber-700">
                Possible misconception: {response.misconception.name}
              </span>
              {response.misconception.clarification && (
                <div className="text-xs text-charcoal-light mt-0.5">
                  Clarification offered: {response.misconception.clarification}
                </div>
              )}
            </div>
          )}
          {response.resolved_misconception && (
            <div className="pt-2 border-t border-zinc-200 text-sm text-emerald-700 font-semibold">
              Misconception resolved: {response.resolved_misconception}
            </div>
          )}
        </div>
        {response.shown_to_student && (
          <div className="text-xs text-charcoal-light">
            <span className="font-semibold text-charcoal">Shown to the student:</span>{' '}
            {response.shown_to_student}
          </div>
        )}
      </div>
    </div>
  )
}

function Row({ label, children }) {
  if (children === null || children === undefined || children === '') return null
  return (
    <div className="flex gap-3 text-sm py-1.5 border-b border-zinc-50 last:border-0">
      <div className="w-44 shrink-0 text-charcoal-light">{label}</div>
      <div className="text-charcoal min-w-0">{children}</div>
    </div>
  )
}

/* The five kinds of information stay in five labelled blocks. They are never
   merged into a single "mastery" number, because they answer different
   questions and can honestly disagree. */
function SessionPanel({ detail, onBack }) {
  const demonstrated = detail.concept_summary.filter((c) => c.status === 'covered')
  const needing = detail.concept_summary.filter((c) => ['partial', 'unclear'].includes(c.status))
  const notDiscussed = detail.concept_summary.filter((c) => c.status === 'not discussed')
  const rels = detail.relationship_summary || []
  const relDemo = rels.filter((r) => r.status === 'demonstrated')
  const relUnclear = rels.filter((r) => r.status === 'needs_clarification')
  const relNone = rels.filter((r) => r.status === 'not_discussed')
  const sr = detail.self_report

  return (
    <div className="space-y-5">
      <div className="card p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-lg font-bold text-charcoal">{detail.student.name}</div>
            <div className="text-sm text-charcoal-light">
              {[detail.student.program, detail.student.roll_no].filter(Boolean).join(' · ')}
            </div>
            <div className="text-sm text-charcoal mt-2">
              {detail.topic.name} · Session {detail.session_id}
              <span className="text-charcoal-light">
                {detail.started_at ? ` · ${new Date(detail.started_at).toLocaleString()}` : ''}
              </span>
            </div>
          </div>
          <div className="flex flex-col items-end gap-2">
            <button onClick={onBack} className="btn-secondary">← All sessions</button>
            {detail.evaluation_closed && <Pill tone="amber">Evaluation closed</Pill>}
          </div>
        </div>
      </div>

      {/* 1. what the student said */}
      <div className="card">
        <div className="card-header">
          <span>Student responses</span>
          <span className="text-white/80 font-normal normal-case">
            {detail.responses_available ? `${detail.responses.length} exchanges` : 'removed'}
          </span>
        </div>
        <div className="p-4 space-y-4">
          {!detail.responses_available && (
            <div className="text-sm bg-amber-50 border border-amber-200 text-amber-900 rounded p-4">
              <div className="font-bold">Raw student responses removed</div>
              <p className="mt-1">
                The evaluation for this lecture was closed
                {detail.evaluation_closed_at ? ` on ${detail.evaluation_closed_at.slice(0, 10)}` : ''},
                and the individual responses were permanently deleted. The evidence drawn from
                them is below and is unaffected.
              </p>
            </div>
          )}
          {detail.responses_available && detail.responses.length === 0 && (
            <p className="text-sm text-charcoal-light">This session has no responses yet.</p>
          )}
          {detail.responses.map((r) => <ResponseCard key={r.exchange_no} response={r} />)}
        </div>
      </div>

      {/* 2. what TeachBack inferred */}
      <div className="card">
        <div className="card-header">What TeachBack concluded</div>
        <div className="p-4">
          <Row label="Concepts demonstrated">
            {demonstrated.length
              ? demonstrated.map((c) => c.name).join(', ')
              : <span className="text-charcoal-light">none</span>}
          </Row>
          <Row label="Needing clarification">
            {needing.length
              ? needing.map((c) => c.name).join(', ')
              : <span className="text-charcoal-light">none</span>}
          </Row>
          <Row label="Not discussed">
            {notDiscussed.length
              ? <span className="text-charcoal-light">
                  {notDiscussed.map((c) => c.name).join(', ')} — no evidence either way
                </span>
              : <span className="text-charcoal-light">none</span>}
          </Row>
          <Row label="Relationships demonstrated">
            {relDemo.length
              ? relDemo.map((r) => `${r.source} → ${r.target}`).join(', ')
              : <span className="text-charcoal-light">none</span>}
          </Row>
          <Row label="Relationships unclear">
            {relUnclear.length
              ? relUnclear.map((r) => `${r.source} → ${r.target}`).join(', ')
              : <span className="text-charcoal-light">none</span>}
          </Row>
          <Row label="Relationships not discussed">
            {relNone.length
              ? <span className="text-charcoal-light">
                  {relNone.map((r) => `${r.source} → ${r.target}`).join(', ')} — no evidence either way
                </span>
              : <span className="text-charcoal-light">none</span>}
          </Row>
          <Row label="Misconceptions detected">
            {detail.misconceptions_detected.length
              ? detail.misconceptions_detected.join(', ')
              : <span className="text-charcoal-light">none</span>}
          </Row>
          <Row label="Misconceptions resolved">
            {detail.misconceptions_resolved.length
              ? detail.misconceptions_resolved.join(', ')
              : <span className="text-charcoal-light">none</span>}
          </Row>
          {(detail.evidence_notes || []).length > 0 && (
            <Row label="Evidence recorded">
              <ul className="list-disc pl-4 space-y-0.5">
                {detail.evidence_notes.map((n, i) => <li key={i}>{n}</li>)}
              </ul>
            </Row>
          )}
        </div>
      </div>

      {/* 3-5. the student's own words about the lecture, the check, the state */}
      <div className="grid lg:grid-cols-2 gap-5">
        <div className="card">
          <div className="card-header">The student&apos;s own account</div>
          <div className="p-4">
            <Row label="Takeaway">
              {detail.takeaway
                ? <span className="italic">“{detail.takeaway}”</span>
                : <span className="text-charcoal-light">
                    {detail.takeaway_removed ? 'removed when the evaluation was closed' : 'none written'}
                  </span>}
            </Row>
            {(detail.summary_insights?.new_concepts_demonstrated || []).length > 0 && (
              <Row label="Added by the takeaway">
                {detail.summary_insights.new_concepts_demonstrated.join(', ')}
              </Row>
            )}
            <Row label="Self-reported">
              {sr
                ? `attention ${sr.attention}/10 · confidence ${sr.confidence}/10 · difficulty ${sr.difficulty}/10`
                : <span className="text-charcoal-light">not reported</span>}
            </Row>
            <Row label="Lecture pace">
              {detail.pace || <span className="text-charcoal-light">not reported</span>}
            </Row>
            <Row label="Requests">
              {(detail.feedback_choices || []).length
                ? detail.feedback_choices.join(' · ')
                : <span className="text-charcoal-light">none</span>}
            </Row>
            <Row label="Written comment">
              {detail.feedback_text
                ? <span className="italic">“{detail.feedback_text}”</span>
                : <span className="text-charcoal-light">
                    {detail.evaluation_closed ? 'removed when the evaluation was closed' : 'none'}
                  </span>}
            </Row>
            <p className="text-[11px] text-charcoal-light pt-3">
              A self-report is what the student said about the session. It is never treated as
              evidence of understanding.
            </p>
          </div>
        </div>

        <div className="card">
          <div className="card-header">Other signals</div>
          <div className="p-4">
            <Row label="Knowledge check">
              {detail.knowledge_check
                ? `${detail.knowledge_check.n_correct}/${detail.knowledge_check.n_questions} correct`
                : <span className="text-charcoal-light">not attempted</span>}
            </Row>
            <Row label="Learning condition">
              {detail.state
                ? <StateBadge label={detail.state.label} />
                : <span className="text-charcoal-light">not estimated</span>}
            </Row>
            <Row label="Recommended next">
              {detail.recommendation?.activity?.title || <span className="text-charcoal-light">none</span>}
            </Row>
            <Row label="Why">
              {detail.recommendation?.why || null}
            </Row>
            <Row label="Activities completed">
              {(detail.activity_completions || []).length
                ? detail.activity_completions.map((c) => c.title).join(', ')
                : <span className="text-charcoal-light">none</span>}
            </Row>
            <p className="text-[11px] text-charcoal-light pt-3">
              The knowledge check is secondary evidence and never rewrites an explanation. The
              learning condition reads this student&apos;s whole session history, not this session
              alone.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Evidence() {
  const context = useTeacherContext()
  const subjectId = context.subject?.id
  const [topics, setTopics] = useState(null)
  const [topicId, setTopicId] = useState(null)
  const [overview, setOverview] = useState(null)
  const [detail, setDetail] = useState(null)
  const [closing, setClosing] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null)

  useEffect(() => {
    if (!subjectId) return
    setTopicId(null)
    setOverview(null)
    setDetail(null)
    api.topics(subjectId).then(setTopics).catch((e) => setMessage({ kind: 'error', text: e.message }))
  }, [subjectId])

  const loadTopic = (id) => {
    if (!subjectId) return
    setDetail(null)
    setTopicId(id)
    setOverview(null)
    api.topicEvidence(id, subjectId).then(setOverview)
      .catch((e) => setMessage({ kind: 'error', text: e.message }))
  }

  const openSession = async (sessionId) => {
    setMessage(null)
    try {
      setDetail(await api.sessionEvidence(sessionId, subjectId))
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    }
  }

  const askClose = async () => {
    setMessage(null)
    try {
      setClosing(await api.closePreview(topicId))
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    }
  }

  const confirmClose = async () => {
    setBusy(true)
    try {
      const r = await api.closeEvaluation(topicId)
      setClosing(null)
      setMessage({ kind: 'ok', text: r.message })
      // re-read everything from the backend: the responses are gone now
      setDetail(null)
      loadTopic(topicId)
      api.topics(subjectId).then(setTopics).catch(() => {})
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-5">
      <div className="banner">Student Evidence — {context.subject?.name || ''}</div>
      <TeacherContextBar context={context} />
      <div className="card p-4 text-sm text-charcoal-light">
        Read what a student actually wrote next to what TeachBack made of it. The system&apos;s
        reading is a suggestion for you to check, not a verdict — pick a lecture, then a student.
      </div>
      {message && (
        <div className={`card p-4 text-sm ${message.kind === 'ok'
          ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
          : 'text-red-700 bg-red-50 border-red-200'}`}>
          {message.text}
        </div>
      )}

      <div className="card">
        <div className="card-header">Lectures in this subject</div>
        <div className="p-4 flex flex-wrap gap-2">
          {!topics && <span className="text-sm text-charcoal-light">Loading…</span>}
          {topics?.length === 0 && (
            <span className="text-sm text-charcoal-light">No topics in {context.subject?.name} yet.</span>
          )}
          {topics?.map((t) => (
            <button
              key={t.id}
              onClick={() => loadTopic(t.id)}
              className={`px-3 py-1.5 rounded-md text-sm font-semibold border transition-colors ${
                t.id === topicId
                  ? 'bg-brand text-white border-brand'
                  : 'bg-white text-charcoal border-zinc-300 hover:border-brand hover:text-brand'
              }`}
            >
              {t.name}
              {t.evaluation_closed && <span className="ml-2 opacity-70">· closed</span>}
            </button>
          ))}
        </div>
      </div>

      {topicId && !overview && <div className="text-sm text-charcoal-light">Loading sessions…</div>}

      {overview && !detail && (
        <div className="card">
          <div className="card-header">
            <span>{overview.topic_name} — TeachBack sessions ({overview.session_count})</span>
          </div>
          <div className="p-4 space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              {overview.evaluation_closed ? (
                <>
                  <Pill tone="amber">Evaluation closed</Pill>
                  <Pill>Raw student responses removed</Pill>
                  <span className="text-xs text-charcoal-light">
                    Closed on {overview.evaluation_closed_at?.slice(0, 10)}. Aggregate learning
                    evidence below is unaffected.
                  </span>
                </>
              ) : (
                <>
                  <Pill tone="emerald">Evaluation open</Pill>
                  <span className="text-xs text-charcoal-light flex-1 min-w-[12rem]">
                    Students can still start this TeachBack, and you can still read their answers.
                  </span>
                  <button
                    onClick={askClose}
                    aria-label={`Close evaluation for ${overview.topic_name}`}
                    className="text-sm font-semibold text-red-700 border border-red-300 hover:bg-red-700 hover:text-white rounded-md px-3 py-1.5 transition-colors"
                  >
                    Close Evaluation
                  </button>
                </>
              )}
            </div>

            {overview.sessions.length === 0 && (
              <p className="text-sm text-charcoal-light">
                No student has done this TeachBack yet.
              </p>
            )}
            {overview.sessions.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-charcoal-light border-b border-zinc-200">
                      <th className="py-2 pr-3">Student</th>
                      <th className="py-2 pr-3">When</th>
                      <th className="py-2 pr-3">Concepts</th>
                      <th className="py-2 pr-3">Learning condition</th>
                      <th className="py-2" />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-100">
                    {overview.sessions.map((s) => (
                      <tr key={s.session_id} className="hover:bg-zinc-50">
                        <td className="py-2.5 pr-3">
                          <div className="font-semibold text-charcoal">{s.student_name}</div>
                          <div className="text-xs text-charcoal-light">
                            {[s.program, s.roll_no].filter(Boolean).join(' · ')}
                          </div>
                        </td>
                        <td className="py-2.5 pr-3 text-xs text-charcoal-light">
                          {s.started_at ? new Date(s.started_at).toLocaleDateString() : '—'}
                          {!s.completed && <div className="text-amber-700">not finished</div>}
                        </td>
                        <td className="py-2.5 pr-3 tabular-nums">
                          {s.concepts_demonstrated}/{s.concepts_total}
                        </td>
                        <td className="py-2.5 pr-3">
                          {s.state_label ? <StateBadge label={s.state_label} /> : '—'}
                        </td>
                        <td className="py-2.5 text-right">
                          <button
                            onClick={() => openSession(s.session_id)}
                            aria-label={`Inspect ${s.student_name}'s session`}
                            className="text-sm font-semibold text-brand border border-brand/40 hover:bg-brand hover:text-white rounded-md px-3 py-1.5 transition-colors whitespace-nowrap"
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {detail && <SessionPanel detail={detail} onBack={() => setDetail(null)} />}

      <CloseEvaluationDialog
        preview={closing}
        topicName={overview?.topic_name || ''}
        busy={busy}
        onCancel={() => setClosing(null)}
        onConfirm={confirmClose}
      />
    </div>
  )
}
