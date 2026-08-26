import { useEffect, useState } from 'react'
import { api } from '../services/api.js'

/* Lightweight teacher/subject context for the faculty interface.
   No authentication — a demo switcher stored in localStorage. The selected
   subject scopes what the lecture and topic management pages show. */

const STORAGE_KEY = 'teachback_teacher_ctx'

export function useTeacherContext() {
  const [teachers, setTeachers] = useState(null)
  const [ctx, setCtx] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}
    } catch {
      return {}
    }
  })
  const [error, setError] = useState(null)

  useEffect(() => {
    api.teachers().then(setTeachers).catch((e) => setError(e.message))
  }, [])

  // resolve the stored ids against the loaded list (fall back to the first)
  let teacher = null
  let subject = null
  if (teachers?.length) {
    teacher = teachers.find((t) => t.id === ctx.teacher_id) || teachers[0]
    subject = teacher.subjects.find((s) => s.id === ctx.subject_id) || teacher.subjects[0] || null
  }

  const select = (teacherId, subjectId) => {
    const t = teachers?.find((x) => x.id === teacherId)
    const next = {
      teacher_id: teacherId,
      subject_id: subjectId ?? (t?.subjects[0]?.id ?? null),
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    setCtx(next)
  }

  return { teachers, teacher, subject, select, error }
}

export function TeacherContextBar({ context }) {
  const { teachers, teacher, subject, select } = context
  if (!teachers) return null
  return (
    <div className="card px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-charcoal-light">Teacher</span>
        <select
          className="input py-1.5 w-auto"
          value={teacher?.id || ''}
          onChange={(e) => select(Number(e.target.value))}
        >
          {teachers.map((t) => (
            <option key={t.id} value={t.id}>{t.name}</option>
          ))}
        </select>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-charcoal-light">Subject</span>
        <select
          className="input py-1.5 w-auto"
          value={subject?.id || ''}
          onChange={(e) => select(teacher.id, Number(e.target.value))}
        >
          {(teacher?.subjects || []).map((s) => (
            <option key={s.id} value={s.id}>{s.name}</option>
          ))}
        </select>
      </div>
      <span className="text-xs text-charcoal-light ml-auto hidden sm:block">
        The selected subject scopes lectures and topics below.
      </span>
    </div>
  )
}
