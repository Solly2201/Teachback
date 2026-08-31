const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail || detail
    } catch {
      /* keep statusText */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),
  students: () => request('/students'),
  student: (id) => request(`/students/${id}`),
  progress: (id) => request(`/students/${id}/progress`),
  topics: (subjectId, includeArchived = false, startableOnly = false) => {
    const params = new URLSearchParams()
    if (subjectId) params.set('subject_id', subjectId)
    if (includeArchived) params.set('include_archived', 'true')
    // students are only offered topics a new session can actually start on
    if (startableOnly) params.set('startable', 'true')
    const query = params.toString()
    return request(`/topics${query ? `?${query}` : ''}`)
  },
  topic: (id) => request(`/topics/${id}`),
  teachers: () => request('/teachers'),
  lectures: (subjectId, includeArchived = false) => {
    const params = new URLSearchParams()
    if (subjectId) params.set('subject_id', subjectId)
    if (includeArchived) params.set('include_archived', 'true')
    const query = params.toString()
    return request(`/lectures${query ? `?${query}` : ''}`)
  },
  lecture: (id) => request(`/lectures/${id}`),
  deletePreview: (id) => request(`/lectures/${id}/delete-preview`),
  deleteLecture: (id) => request(`/lectures/${id}`, { method: 'DELETE' }),
  restoreLecture: (id) => request(`/lectures/${id}/restore`, { method: 'POST' }),
  createLecture: (data) => request('/lectures', { method: 'POST', body: JSON.stringify(data) }),
  updateLecture: (id, data) => request(`/lectures/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  publishLecture: (id) => request(`/lectures/${id}/publish`, { method: 'POST' }),
  extractMaterial: (filename, contentBase64) =>
    request('/lectures/extract', { method: 'POST', body: JSON.stringify({ filename, content_base64: contentBase64 }) }),
  createTopic: (data) => request('/topics', { method: 'POST', body: JSON.stringify(data) }),
  updateTopic: (id, data) => request(`/topics/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  topicDeletePreview: (id) => request(`/topics/${id}/delete-preview`),
  deleteTopic: (id) => request(`/topics/${id}`, { method: 'DELETE' }),
  restoreTopic: (id) => request(`/topics/${id}/restore`, { method: 'POST' }),
  closePreview: (id) => request(`/topics/${id}/close-preview`),
  closeEvaluation: (id) => request(`/topics/${id}/close-evaluation`, { method: 'POST' }),
  // teacher oversight: the subject is always sent, so a topic or session id
  // can never reach another teacher's students
  topicEvidence: (topicId, subjectId) =>
    request(`/teacher/topics/${topicId}/evidence?subject_id=${subjectId}`),
  sessionEvidence: (sessionId, subjectId) =>
    request(`/teacher/sessions/${sessionId}/evidence?subject_id=${subjectId}`),
  startSession: (student_id, topic_id) =>
    request('/sessions/start', { method: 'POST', body: JSON.stringify({ student_id, topic_id }) }),
  respond: (sessionId, text) =>
    request(`/sessions/${sessionId}/respond`, { method: 'POST', body: JSON.stringify({ text }) }),
  finish: (sessionId, report) =>
    request(`/sessions/${sessionId}/finish`, { method: 'POST', body: JSON.stringify(report) }),
  topicQuiz: (topicId) => request(`/topics/${topicId}/quiz`),
  submitQuiz: (quizId, data) => request(`/quiz/${quizId}/submit`, { method: 'POST', body: JSON.stringify(data) }),
  regenerateQuiz: (lectureId, index) =>
    request(`/lectures/${lectureId}/quiz/regenerate`, { method: 'POST', body: JSON.stringify({ index }) }),
  activity: (id) => request(`/activities/${id}`),
  completeActivity: (data) => request('/activities/complete', { method: 'POST', body: JSON.stringify(data) }),
  teacherOverview: (subjectId) => request(`/teacher/overview${subjectId ? `?subject_id=${subjectId}` : ''}`),
  prepPrompt: () => request('/lectures/prep-prompt'),
  metaStates: () => request('/meta/states'),
  evaluation: () => request('/meta/evaluation'),
}

export const STATE_META = {
  'Not Trying': { color: 'bg-zinc-200 text-zinc-700 border-zinc-300', dot: '#71717A', icon: '○' },
  'Unclear': { color: 'bg-amber-50 text-amber-800 border-amber-200', dot: '#D97706', icon: '◔' },
  'Struggling but Trying': { color: 'bg-orange-50 text-orange-800 border-orange-200', dot: '#EA580C', icon: '◑' },
  'Understanding': { color: 'bg-sky-50 text-sky-800 border-sky-200', dot: '#0369A1', icon: '◕' },
  'Confident': { color: 'bg-emerald-50 text-emerald-800 border-emerald-200', dot: '#047857', icon: '●' },
}

export const STATE_ORDER = ['Not Trying', 'Unclear', 'Struggling but Trying', 'Understanding', 'Confident']

export const STATE_DESCRIPTIONS = {
  'Not Trying': 'Low engagement in recent sessions — a short re-engagement activity is the next step.',
  'Unclear': 'The core ideas are not settled yet — a simpler explanation will help.',
  'Struggling but Trying': 'Strong effort with gaps in accuracy — guided practice is most effective here.',
  'Understanding': 'The concepts are coming together — time to apply them.',
  'Confident': 'Consistently accurate explanations — ready for advanced challenges.',
}

/* Student-facing wording for the same five states. The internal model is
   unchanged and faculty views keep the formal names, but a student is never
   told something the system cannot observe: TeachBack sees answers and
   self-reports, never intent. "Not Trying" therefore becomes a statement
   about the evidence, not about the person. */
export const STATE_STUDENT_LABELS = {
  'Not Trying': 'Very little evidence yet',
  'Unclear': 'Still unclear',
  'Struggling but Trying': 'Working through it',
  'Understanding': 'Understanding',
  'Confident': 'Confident',
}

export const STATE_STUDENT_DESCRIPTIONS = {
  'Not Trying': "Your recent sessions didn't give us much to go on yet — one short answer is enough to change that.",
  'Unclear': "The core ideas haven't settled yet — a simpler explanation will help more than practice right now.",
  'Struggling but Trying': "You're putting in real effort and some parts are still coming together — guided practice fits best here.",
  'Understanding': 'The concepts are coming together — the next step is applying them.',
  'Confident': "You've explained these ideas clearly across recent sessions — ready for an optional challenge.",
}

export const studentStateLabel = (label) => STATE_STUDENT_LABELS[label] || label
export const studentStateDescription = (label) =>
  STATE_STUDENT_DESCRIPTIONS[label] || STATE_DESCRIPTIONS[label] || ''
