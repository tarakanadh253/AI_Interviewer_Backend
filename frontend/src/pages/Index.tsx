import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Clock, Target, Award, Sparkles } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Clock,
      title: "30 Minutes",
      description: "Quick, focused interview sessions designed to fit your schedule.",
      gradient: "from-blue-500/20 to-cyan-500/20",
      text: "text-blue-400",
    },
    {
      icon: Target,
      title: "5 Topics",
      description: "Choose from multiple tech domains including Python, System Design, and more.",
      gradient: "from-purple-500/20 to-pink-500/20",
      text: "text-purple-400",
    },
    {
      icon: Award,
      title: "1 Free Trial",
      description: "One attempt per Google account to evaluate your skills.",
      gradient: "from-amber-500/20 to-orange-500/20",
      text: "text-amber-400",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 flex flex-col items-center justify-center p-6 relative overflow-hidden">
      {/* Background Gradients */}
      <div className="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-blue-900/20 to-transparent pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-indigo-900/10 rounded-full blur-[100px] pointer-events-none" />

      <div className="max-w-6xl w-full relative z-10">
        {/* Hero Section */}
        <div className="text-center mb-24 animate-fade-in">
          <div className="inline-flex items-center justify-center p-3 mb-8 bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-full shadow-lg">
            <Sparkles className="h-5 w-5 text-indigo-400 mr-2" />
            <span className="text-sm font-medium text-slate-300">New: Enhanced AI Evaluation Model</span>
          </div>

          <h1 className="text-6xl md:text-8xl font-bold mb-8 tracking-tight text-white">
            Master Your <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400">
              Tech Interview
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed">
            Practice with our advanced AI voice interviewer. Get real-time feedback on your communication and technical skills.
          </p>

          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
            <Button
              onClick={() => navigate("/login")}
              className="px-12 py-8 text-xl bg-white text-slate-950 hover:bg-slate-200 rounded-full font-semibold shadow-[0_0_20px_-5px_rgba(255,255,255,0.3)] transition-all duration-300 hover:scale-105"
            >
              Start Interview
            </Button>
            <Button
              variant="outline"
              onClick={() => navigate("/results?demo=true")}
              className="px-12 py-8 text-xl bg-slate-900/50 border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white rounded-full transition-all duration-300"
            >
              View Sample Results
            </Button>
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mb-20">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card
                key={index}
                className="group p-8 bg-slate-900/40 backdrop-blur-xl border-slate-800 hover:border-slate-700 transition-all duration-500 hover:bg-slate-900/60"
              >
                <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-br ${feature.gradient} mb-6 group-hover:scale-110 transition-transform duration-500`}>
                  <Icon className={`h-8 w-8 ${feature.text}`} />
                </div>
                <h3 className="text-2xl font-semibold text-white mb-4">
                  {feature.title}
                </h3>
                <p className="text-slate-400 leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            );
          })}
        </div>

        {/* Info Banner */}
        <div className="max-w-3xl mx-auto">
          <div className="p-1 rounded-2xl bg-gradient-to-r from-slate-800 to-slate-900">
            <div className="rounded-xl bg-slate-950 p-6 flex items-center justify-center gap-4">
              <span className="text-2xl">💡</span>
              <p className="text-slate-400 text-lg">
                <span className="text-white font-medium">Pro Tip:</span> Speak naturally. Our AI analyzes both your content and delivery style.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
