import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'

function fmtTs(s) {
  const total = Math.floor(s)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const sec = total % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  return `${m}:${String(sec).padStart(2, '0')}`
}

export default function TroubleDensityChart({ conceptReports, unmatchedComments, duration }) {
  const data = useMemo(() => {
    const bucketSize = Math.max(60, Math.ceil(duration / 30))
    const numBuckets = Math.ceil(duration / bucketSize)
    const counts = Array(numBuckets).fill(0)

    conceptReports.forEach(report => {
      report.matched_comments?.forEach(mc => {
        mc.timestamps_in_text?.forEach(ts => {
          const b = Math.min(Math.floor(ts / bucketSize), numBuckets - 1)
          counts[b]++
        })
      })
    })

    unmatchedComments?.forEach(mc => {
      mc.timestamps_in_text?.forEach(ts => {
        const b = Math.min(Math.floor(ts / bucketSize), numBuckets - 1)
        counts[b]++
      })
    })

    return counts.map((count, i) => ({
      time: i * bucketSize,
      count,
      label: fmtTs(i * bucketSize),
    }))
  }, [conceptReports, unmatchedComments, duration])

  if (data.every(d => d.count === 0)) {
    return <p style={{ color: '#b2bec3' }}>No trouble comments with timestamps to display.</p>
  }

  const tickInterval = Math.max(0, Math.ceil(data.length / 8) - 1)

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 35 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="label"
          interval={tickInterval}
          label={{ value: 'Video timestamp', position: 'insideBottom', offset: -20, fontSize: 12 }}
          tick={{ fontSize: 11 }}
        />
        <YAxis allowDecimals={false} tick={{ fontSize: 11 }} width={30} />
        <Tooltip
          formatter={(v) => [v, 'Trouble comments']}
          labelFormatter={(label) => `Around ${label}`}
        />
        <Bar dataKey="count" fill="#e17055" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
