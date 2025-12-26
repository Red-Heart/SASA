"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Loader2, Thermometer, Droplets, CloudRain, Beaker, MapPin, Sprout, Leaf } from "lucide-react"

// Configuration
const API_BASE_URL = "http://localhost:8000"

const CROPS = [
  "rice",
  "maize",
  "chickpea",
  "kidneybeans",
  "pigeonpeas",
  "mothbeans",
  "mungbean",
  "blackgram",
  "lentil",
  "pomegranate",
  "banana",
  "mango",
  "grapes",
  "watermelon",
  "muskmelon",
  "apple",
  "orange",
  "papaya",
  "coconut",
  "cotton",
  "jute",
  "coffee",
]

export default function AgricultureAdvisoryPage() {
  const [location, setLocation] = useState("")
  const [selectedCrop, setSelectedCrop] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<any>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/advisory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location,
          crop: selectedCrop === "auto" ? null : selectedCrop,
        }),
      })

      if (!response.ok) throw new Error("Failed to fetch advisory data")

      const result = await response.json()
      setData(result)
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred")
    } finally {
      setLoading(false)
    }
  }

  const getIrrigationColor = (level: string) => {
    switch (level.toLowerCase()) {
      case "low":
        return "text-green-400 bg-green-400/10 border-green-400/20"
      case "medium":
        return "text-yellow-400 bg-yellow-400/10 border-yellow-400/20"
      case "high":
        return "text-red-400 bg-red-400/10 border-red-400/20"
      default:
        return "text-muted-foreground"
    }
  }

  return (
    <main className="min-h-screen pb-20 max-w-7xl mx-auto space-y-12">
      <section className="relative pt-16 pb-12 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full -z-10">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-green-900/10 rounded-full blur-3xl animate-pulse delay-700" />
        </div>

        <div className="space-y-4 text-center px-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-widest animate-in fade-in slide-in-from-bottom-4 duration-1000">
            <Leaf className="w-3 h-3" /> AI-Powered Farming Advice
          </div>
          <h1 className="text-4xl md:text-6xl font-black tracking-tight bg-linear-to-b from-foreground to-foreground/70 bg-clip-text text-transparent animate-in fade-in slide-in-from-bottom-6 duration-1000 delay-100">
            Smart Agriculture <span className="text-primary">Advisory System</span>
          </h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-2xl mx-auto text-balance animate-in fade-in slide-in-from-bottom-8 duration-1000 delay-200">
            Get real-time Irrigation advice and Crop recommendations, along with predicted yield potential based on your location.
          </p>
        </div>
      </section>

      <section className="grid lg:grid-cols-3 gap-8 px-4">
        {/* Input Form */}
        <Card className="lg:col-span-1 bg-card/40 backdrop-blur-xl border-gray-600 shadow-2xl ">
          <CardHeader>
            <CardTitle>Advisory Parameters</CardTitle>
            <CardDescription>Enter details to get intelligent farming advice</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-primary" /> Location
                </label>
                <Input
                  placeholder="Location in Malaysia"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  required
                  className="bg-background/50"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Sprout className="w-4 h-4 text-primary" /> Crop Focus
                </label>
                <Select
                  value={selectedCrop ?? "auto"}
                  onValueChange={(val) => setSelectedCrop(val === "auto" ? null : val)}
                >
                  <SelectTrigger className="h-11 bg-background/40 border border-primary/20 hover:border-primary/40 transition-colors">
                    <SelectValue placeholder="Auto-select best crops" />
                  </SelectTrigger>

                  <SelectContent
                    position="popper"
                    sideOffset={4}
                    className="
                      z-50
                      max-h-64
                      w-[--radix-select-trigger-width]
                      overflow-y-auto
                      rounded-md
                      border border-primary/20
                      bg-card
                      shadow-lg
                    "
                  >
                    <SelectItem
                      value="auto"
                      className="
                        cursor-pointer
                        px-3 py-2
                        data-highlighted:bg-primary/15
                        data-highlighted:text-primary
                        data-[state=checked]:font-semibold
                      "
                    >
                      Auto-select best crops
                    </SelectItem>

                    {CROPS.map((crop) => (
                      <SelectItem
                        key={crop}
                        value={crop}
                        className="
                          cursor-pointer capitalize
                          px-3 py-2
                          data-highlighted:bg-primary/15
                          data-highlighted:text-primary
                          data-[state=checked]:font-semibold
                        "
                      >
                        {crop}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

              </div>
              <Button type="submit" className="w-full mt-4" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Fetching...
                  </>
                ) : (
                  "Get Advisory"
                )}
              </Button>
            </form>
            {error && (
              <p className="mt-4 text-sm text-destructive font-medium p-3 rounded-md bg-destructive/10 border border-destructive/20">
                {error}
              </p>
            )}
          </CardContent>
        </Card>

        {/* System Factors & Irrigation */}
        <div className="lg:col-span-2 space-y-8">
          {data && (
            <>
              {/* Environmental Factors */}
              <div className="space-y-4">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-primary" /> System-derived environmental factors
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <FactorCard
                    icon={<Thermometer className="text-orange-400" />}
                    label="Temp"
                    value={`${data.environmental_factors.temperature.toFixed(1)}°C`}
                  />
                  <FactorCard
                    icon={<Droplets className="text-blue-400" />}
                    label="Humidity"
                    value={`${data.environmental_factors.humidity.toFixed(1)}%`}
                  />
                  <FactorCard
                    icon={<CloudRain className="text-cyan-400" />}
                    label="Rainfall"
                    value={`${data.environmental_factors.rainfall.toFixed(1)}mm`}
                  />
                  <FactorCard
                    icon={<Beaker className="text-green-400" />}
                    label="pH"
                    value={data.environmental_factors.ph.toFixed(1)}
                  />
                  <FactorCard label="Nitrogen (N)" value={data.environmental_factors.N.toFixed(1)} subValue="ppm" />
                  <FactorCard label="Phosphorus (P)" value={data.environmental_factors.P.toFixed(1)} subValue="ppm" />
                  <FactorCard label="Potassium (K)" value={data.environmental_factors.K.toFixed(1)} subValue="ppm" />
                </div>
              </div>

              {/* Irrigation Advice */}
              <Card className="border-primary/10 overflow-hidden">
                <div className="flex flex-col md:flex-row items-center gap-6 p-6">
                  <div className="space-y-1 text-center md:text-left">
                    <h3 className="text-lg font-semibold">Irrigation Strategy</h3>
                    <p className="text-sm text-muted-foreground">Fuzzy Logic based water management</p>
                  </div>
                  <div className="flex-1 flex gap-4 justify-center md:justify-end w-full">
                    <div
                      className={`px-6 py-3 rounded-xl border-2 flex flex-col items-center justify-center min-w-35 ${getIrrigationColor(data.advisory.irrigation.irrigation_level)}`}
                    >
                      <span className="text-xs font-bold uppercase tracking-widest opacity-70">Level</span>
                      <span className="text-xl font-black">{data.advisory.irrigation.irrigation_level}</span>
                    </div>
                    <div className="px-6 py-3 rounded-xl border-2 border-border bg-muted/50 flex flex-col items-center justify-center min-w-35">
                      <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Score</span>
                      <span className="text-xl font-black text-primary">
                        {data.advisory.irrigation.irrigation_score.toFixed(1)}
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            </>
          )}

          {!data && !loading && (
            <div className="h-full flex items-center justify-center border-2 border-dashed border-gray-600 rounded-xl p-12 text-center text-muted-foreground">
              <div className="space-y-4">
                <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center mx-auto">
                  <Sprout className="w-6 h-6" />
                </div>
                <p>Submit location data to receive environmental analysis and crop advisory.</p>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Advisory Results */}
      {data && (
        <section className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold tracking-tight">
              {data.advisory.mode === "crop_specific" ? "Crop Suitability Analysis" : "Top Recommended Crops"}
            </h2>
            <Badge
              variant="outline"
              className="text-xs uppercase tracking-widest bg-primary/5 border-primary/20 text-primary"
            >
              Mode: {data.advisory.mode.replace("_", " ")}
            </Badge>
          </div>

          {data.advisory.mode === "crop_specific" ? (
            <Card className="border-primary/20 bg-linear-to-br from-card to-background overflow-hidden">
              <div className="grid md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-border">
                <div className="p-8 space-y-4">
                  <div className="space-y-1">
                    <h3 className="text-sm font-bold uppercase tracking-tighter text-muted-foreground">Target Crop</h3>
                    <p className="text-4xl font-black text-primary capitalize">{data.advisory.crop}</p>
                  </div>
                  <div className="pt-4 space-y-4">
                    <ScoreDisplay label="Suitability" score={data.advisory.suitability_score} />
                    <ScoreDisplay label="Yield Potential" score={data.advisory.yield_potential} />
                  </div>
                </div>
                <div className="p-8 md:col-span-2 space-y-6 bg-muted/30">
                  <div className="space-y-2">
                    <h3 className="text-lg font-bold">Planting Context</h3>
                    <div className="flex items-center gap-3">
                      <Badge className="text-lg px-4 py-1 font-mono bg-primary text-primary-foreground">
                        {data.advisory.planting_context.context_code}
                      </Badge>
                      <span className="text-xl font-medium">{data.advisory.planting_context.context_label}</span>
                    </div>
                  </div>
                  <div className="grid md:grid-cols-2 gap-8">
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground uppercase font-bold tracking-widest">Current Month</p>
                      <p className="text-2xl font-bold">
                        {new Intl.DateTimeFormat("en-US", { month: "long" }).format(
                          new Date(2025, data.advisory.planting_context.current_month - 1),
                        )}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground uppercase font-bold tracking-widest">
                        Typical Planting Window
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {data.advisory.planting_context.usual_planting_months.map((m: number) => (
                          <Badge key={m} variant="secondary" className="bg-secondary/50">
                            {new Intl.DateTimeFormat("en-US", { month: "short" }).format(new Date(2025, m - 1))}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4">
              {data.advisory.top_5_crops.map((item: any, idx: number) => (
                <Card
                  key={idx}
                  className="bg-card/40 border-primary/10 hover:border-primary/30 transition-colors group"
                >
                  <CardContent className="p-4 space-y-4">
                    <div className="flex justify-between items-start">
                      <p className="text-xl font-bold capitalize text-primary">{item.crop}</p>
                      <Badge variant="outline" className="font-mono text-[10px]">
                        {item.planting_context.context_code}
                      </Badge>
                    </div>
                    <div className="space-y-3">
                      <CompactScore label="Suitability" score={item.suitability_score} />
                      <CompactScore label="Yield" score={item.yield_potential} />
                    </div>
                    <p className="text-[10px] text-muted-foreground leading-tight italic line-clamp-2">
                      {item.planting_context.context_label}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  )
}

function FactorCard({
  icon,
  label,
  value,
  subValue,
}: { icon?: React.ReactNode; label: string; value: string | number; subValue?: string }) {
  return (
    <div className="bg-muted/40 border border-border/50 p-4 rounded-xl flex flex-col gap-1 transition-all hover:bg-muted/60">
      <div className="flex items-center gap-2 text-xs font-bold text-muted-foreground uppercase tracking-widest">
        {icon} {label}
      </div>
      <div className="text-xl font-black flex items-baseline gap-1">
        {value}
        {subValue && <span className="text-xs font-normal opacity-50">{subValue}</span>}
      </div>
    </div>
  )
}

function ScoreDisplay({ label, score }: { label: string; score: number }) {
  const percentage = score * 100
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs font-bold uppercase tracking-widest">
        <span>{label}</span>
        <span className="text-primary">{percentage.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden border border-border/50">
        <div className="h-full bg-primary transition-all duration-500 ease-out" style={{ width: `${percentage}%` }} />
      </div>
    </div>
  )
}

function CompactScore({ label, score }: { label: string; score: number }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] font-bold uppercase text-muted-foreground">
        <span>{label}</span>
        <span>{(score * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1 bg-muted rounded-full overflow-hidden">
        <div className="h-full bg-primary" style={{ width: `${score * 100}%` }} />
      </div>
    </div>
  )
}