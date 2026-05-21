import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const COLORS = ['#2ecc71', '#95a5a6', '#e74c3c']

export default function SentimentChart({ sentiment }) {
  const data = [
    { name: 'Positive', value: sentiment.positive_pct },
    { name: 'Neutral',  value: sentiment.neutral_pct  },
    { name: 'Negative', value: sentiment.negative_pct },
  ]

  return (
    <ResponsiveContainer width="100%" height={280}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={70}
          outerRadius={110}
          dataKey="value"
          label={({ name, value }) => `${name}: ${value}%`}
          labelLine={true}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i]} />
          ))}
        </Pie>
        <Tooltip formatter={(v) => `${v}%`} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  )
}
