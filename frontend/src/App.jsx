import { useEffect, useState } from 'react'
import { Navigate, Route, Routes, useNavigate } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import Activity from './pages/Activity.jsx'
import Evidence from './pages/Evidence.jsx'
import Progress from './pages/Progress.jsx'
import RoleSelect from './pages/RoleSelect.jsx'
import StudentDashboard from './pages/StudentDashboard.jsx'
import TeachBack from './pages/TeachBack.jsx'
import Lectures from './pages/Lectures.jsx'
import TeacherDashboard from './pages/TeacherDashboard.jsx'
import Topics from './pages/Topics.jsx'

export default function App() {
  const [user, setUser] = useState(() => {
    const raw = localStorage.getItem('teachback_user')
    return raw ? JSON.parse(raw) : null
  })
  const navigate = useNavigate()

  useEffect(() => {
    if (user) localStorage.setItem('teachback_user', JSON.stringify(user))
    else localStorage.removeItem('teachback_user')
  }, [user])

  const signOut = () => {
    setUser(null)
    navigate('/')
  }

  if (!user) {
    return (
      <Routes>
        <Route path="*" element={<RoleSelect onSelect={setUser} />} />
      </Routes>
    )
  }

  return (
    <Layout user={user} onSignOut={signOut}>
      <Routes>
        {user.role === 'student' ? (
          <>
            <Route path="/" element={<StudentDashboard user={user} />} />
            <Route path="/teachback" element={<TeachBack user={user} />} />
            <Route path="/activity" element={<Activity user={user} />} />
            <Route path="/progress" element={<Progress user={user} />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </>
        ) : (
          <>
            <Route path="/" element={<TeacherDashboard />} />
            <Route path="/lectures" element={<Lectures />} />
            <Route path="/topics" element={<Topics />} />
            <Route path="/evidence" element={<Evidence />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </>
        )}
      </Routes>
    </Layout>
  )
}
