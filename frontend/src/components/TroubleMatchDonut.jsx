import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const COLORS = ['#00b894', '#b2bec3']

export default function TroubleMatchDonut({ troubleCount, unmatchedCount }) {
  const matched = troubleCount - unmatchedCount
  const data = [
    { name: 'Mapped to concept', value: matched },
    { name: 'Unmatched', value: unmatchedCount },
  ].filter(d => d.value > 0)

  if (data.length === 0) {
    return <p style={{ color: '#b2bec3' }}>No trouble comments.</p>
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={90}
          dataKey="value"
          paddingAngle={3}
        >
          {data.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
        </Pie>
        <Tooltip formatter={(v, name) => [v, name]} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  )
}
