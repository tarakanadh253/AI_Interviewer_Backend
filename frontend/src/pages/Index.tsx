import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Clock, Target, Award, Sparkles } from "lucide-react";
import { Navbar } from "@/components/Navbar";

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Clock,
      title: "30 Minutes",
      description: "Quick, focused interview sessions designed to fit your schedule.",
      gradient: "from-ohg-navy/10 to-ohg-teal/10",
      text: "text-ohg-navy",
    },
    {
      icon: Target,
      title: "5 Topics",
      description: "Choose from multiple tech domains including Python, System Design, and more.",
      gradient: "from-ohg-orange/10 to-pink-500/10",
      text: "text-ohg-orange",
    },
    {
      icon: Award,
      title: "1 Free Trial",
      description: "One attempt per Google account to evaluate your skills.",
      gradient: "from-ohg-teal/10 to-cyan-500/10",
      text: "text-ohg-teal",
    },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col relative overflow-x-hidden">
      {/* Navbar */}
      <Navbar />

      {/* Background Gradients */}
      <div className="fixed top-0 right-0 w-[800px] h-[800px] bg-ohg-navy/5 rounded-full blur-[100px] pointer-events-none -translate-y-1/2 translate-x-1/2 z-0" />
      <div className="fixed bottom-0 left-0 w-[800px] h-[800px] bg-ohg-orange/5 rounded-full blur-[100px] pointer-events-none translate-y-1/2 -translate-x-1/2 z-0" />

      {/* Main Content */}
      <main className="flex-grow flex flex-col items-center justify-center p-6 relative z-10 pt-10 md:pt-20">
        <div className="max-w-6xl w-full">
          {/* Hero Section */}
          <div className="text-center mb-20 animate-fade-in">
            <div className="inline-flex items-center justify-center p-3 mb-8 bg-white/50 dark:bg-white/10 backdrop-blur-xl border border-border rounded-full shadow-sm">
              <Sparkles className="h-5 w-5 text-ohg-teal mr-2" />
              <span className="text-sm font-medium text-ohg-navy dark:text-gray-200">New: Enhanced AI Evaluation Model</span>
            </div>

            <h1 className="text-5xl md:text-8xl font-bold mb-8 tracking-tight text-ohg-navy-light">
              Master Your <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-ohg-orange to-ohg-teal">
                Tech Interview
              </span>
            </h1>

            <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-2xl mx-auto leading-relaxed">
              Practice with our advanced AI voice interviewer. Get real-time feedback on your communication and technical skills.
            </p>

            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <Button
                onClick={() => navigate("/login")}
                className="px-12 py-8 text-xl bg-ohg-orange text-white hover:bg-ohg-orange-hover rounded-full font-bold shadow-xl hover:shadow-ohg-orange/30 transition-all duration-300 hover:scale-105"
              >
                Start Interview
              </Button>
              <Button
                variant="outline"
                onClick={() => navigate("/results?demo=true")}
                className="px-12 py-8 text-xl bg-white dark:bg-transparent border-border text-ohg-navy dark:text-white hover:bg-gray-50 dark:hover:bg-white/10 hover:text-ohg-navy rounded-full transition-all duration-300 font-medium"
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
                  className="group p-8 bg-white/60 dark:bg-white/5 backdrop-blur-xl border-border hover:border-ohg-teal/30 transition-all duration-500 hover:bg-white dark:hover:bg-white/10 hover:shadow-xl hover:-translate-y-1"
                >
                  <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-br ${feature.gradient} mb-6 group-hover:scale-110 transition-transform duration-500`}>
                    <Icon className={`h-8 w-8 ${feature.text}`} />
                  </div>
                  <h3 className="text-2xl font-bold text-ohg-navy dark:text-white mb-4">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </Card>
              );
            })}
          </div>

          {/* Info Banner */}
          <div className="max-w-3xl mx-auto mb-20">
            <div className="p-1 rounded-2xl bg-gradient-to-r from-ohg-navy/10 to-ohg-teal/10">
              <div className="rounded-xl bg-white dark:bg-black/40 p-6 flex items-center justify-center gap-4 border border-border shadow-sm">
                <span className="text-2xl">💡</span>
                <p className="text-muted-foreground text-lg">
                  <span className="text-ohg-navy dark:text-white font-bold">Pro Tip:</span> Speak naturally. Our AI analyzes both your content and delivery style.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
