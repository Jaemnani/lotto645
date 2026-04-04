interface LottoBallProps {
  number: number
  size?: 'sm' | 'md' | 'lg'
  isBonus?: boolean
  highlight?: boolean
}

function getBallColor(n: number): string {
  if (n <= 10) return 'bg-yellow-400 text-yellow-900'
  if (n <= 20) return 'bg-blue-500 text-white'
  if (n <= 30) return 'bg-red-500 text-white'
  if (n <= 40) return 'bg-gray-600 text-white'
  return 'bg-green-500 text-white'
}

const sizeClass = {
  sm: 'w-8 h-8 text-xs font-bold',
  md: 'w-11 h-11 text-sm font-bold',
  lg: 'w-14 h-14 text-base font-bold',
}

export default function LottoBall({
  number,
  size = 'md',
  isBonus = false,
  highlight = false,
}: LottoBallProps) {
  return (
    <div
      className={[
        'rounded-full flex items-center justify-center select-none shadow-md',
        sizeClass[size],
        isBonus
          ? 'ring-2 ring-offset-1 ring-purple-400 bg-purple-500 text-white'
          : getBallColor(number),
        highlight ? 'ring-2 ring-offset-1 ring-white scale-110' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {number}
    </div>
  )
}
