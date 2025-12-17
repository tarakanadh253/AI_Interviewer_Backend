import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Download, Home, Trophy, Target, Lightbulb, Loader2 } from "lucide-react";
import jsPDF from "jspdf";
import { useToast } from "@/hooks/use-toast";
import { apiService, type InterviewSession } from "@/lib/api";

const Results = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadResults = async () => {
      // Check for demo mode
      const searchParams = new URLSearchParams(window.location.search);
      const isDemo = searchParams.get('demo') === 'true';

      if (isDemo) {
        // Load mock data for sample view
        setSession({
          id: 999,
          user: 0,
          user_email: "demo@example.com",
          user_name: "Demo User",
          started_at: new Date().toISOString(),
          ended_at: new Date().toISOString(),
          duration_seconds: 1200,
          topics: [],
          topics_list: [
            { id: 1, name: "Python", question_count: 5 },
            { id: 2, name: "System Design", question_count: 5 }
          ],
          status: 'COMPLETED',
          communication_score: 0.85,
          technology_score: 0.92,
          result_summary: "Strong performance. Improvements: Consider discussing trade-offs more deeply, Mention scalability challenges earlier",
          answer_count: 10,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          answers: Array(5).fill(null).map((_, i) => ({
            id: i,
            session: 999,
            question: i,
            question_text: "Explain the Global Interpreter Lock (GIL) in Python and its impact on multi-threaded programs.",
            user_answer: "The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. This means even on multi-core processors, only one thread executes at a time.",
            similarity_score: 0.88,
            accuracy_score: 0.9,
            completeness_score: 0.85,
            communication_subscore: 0.9,
            matched_keywords: "['mutex', 'threads', 'bytecode', 'multi-core']",
            missing_keywords: "['CPython', 'memory management']",
            topic_score: 0.9,
            score_breakdown: "",
            created_at: new Date().toISOString()
          }))
        } as any);
        setIsLoading(false);
        return;
      }

      try {
        const sessionId = localStorage.getItem('completed_session_id') || localStorage.getItem('session_id');
        if (!sessionId) {
          toast({
            title: "No Results Found",
            description: "Please complete an interview first.",
            variant: "destructive",
          });
          navigate("/");
          return;
        }

        const sessionData = await apiService.getSessionResults(parseInt(sessionId));
        setSession(sessionData);
      } catch (error: any) {
        console.error('Error loading results:', error);
        toast({
          title: "Error",
          description: error.message || "Could not load results. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadResults();
  }, [navigate, toast]);

  // Calculate scores
  const answeredCount = session?.answer_count || (session?.answers?.length || 0);

  // Calculate total questions based on topics
  // The interview selects up to 10 questions from the available pool
  const maxPotentialQuestions = session?.topics_list?.reduce((sum: number, t: any) => sum + (t.question_count || 0), 0) || 0;
  // If we can't determine it (e.g. old session), default to 10 or answered count if higher
  const totalQuestions = maxPotentialQuestions > 0 ? Math.min(maxPotentialQuestions, 10) : Math.max(answeredCount, 10);

  const completionPercentage = totalQuestions > 0 ? (answeredCount / totalQuestions) : 0;

  // Raw scores from answered questions (0-100)
  const rawCommunicationScore = session?.communication_score ? Math.round(session.communication_score * 100) : 0;
  const rawTechnicalScore = session?.technology_score ? Math.round(session.technology_score * 100) : 0;

  // Calculate overall score weighted by completion
  // Unanswered questions count as 0% towards the goal
  const communicationScore = rawCommunicationScore;
  const technicalScore = rawTechnicalScore;

  // Overall score penalizes for missed questions
  const overallScore = Math.round(((communicationScore + technicalScore) / 2) * completionPercentage);

  // Parse improvements from result_summary
  const improvements: string[] = [];
  if (session?.result_summary) {
    const improvementsMatch = session.result_summary.match(/Improvements:\s*(.+)/i);
    if (improvementsMatch) {
      improvements.push(...improvementsMatch[1].split(',').map(i => i.trim()));
    }
  }

  const handleDownload = () => {
    if (!session) return;

    try {
      const doc = new jsPDF();

      doc.setFontSize(20);
      doc.text("Interview Results Report", 20, 20);

      doc.setFontSize(16);
      doc.text(`Overall Score: ${overallScore}%`, 20, 40);

      doc.setFontSize(14);
      doc.text("Score Breakdown:", 20, 55);

      doc.setFontSize(12);
      doc.text(`Communication: ${communicationScore}%`, 30, 70);
      doc.text(`Technical Skills: ${technicalScore}%`, 30, 85);

      if (improvements.length > 0) {
        doc.setFontSize(14);
        doc.text("Areas for Improvement:", 20, 105);

        doc.setFontSize(10);
        improvements.forEach((improvement, index) => {
          doc.text(`${index + 1}. ${improvement}`, 30, 115 + (index * 15));
        });
      }

      doc.save("interview-results.pdf");

      toast({
        title: "Report downloaded",
        description: "Your interview results have been saved as PDF",
      });
    } catch (error) {
      console.error("PDF generation error:", error);
      toast({
        title: "Download failed",
        description: "Could not generate PDF report",
        variant: "destructive",
      });
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="text-center">
          <Loader2 className="h-10 w-10 animate-spin mx-auto mb-4 text-blue-500" />
          <p className="text-slate-400">Analyzing performance...</p>
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 p-6">
        <Card className="p-10 max-w-md w-full bg-slate-900 border-slate-800 text-center shadow-2xl rounded-3xl">
          <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-6">
            <Trophy className="h-8 w-8 text-slate-500" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">No Results Found</h2>
          <p className="text-slate-400 mb-8">
            Complete an interview to see your detailed analysis here.
          </p>
          <Button
            onClick={() => navigate("/")}
            className="w-full bg-white text-slate-950 hover:bg-slate-200 h-12 rounded-xl font-semibold"
          >
            Return Home
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 relative overflow-hidden">
      {/* Background Ambience */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-blue-900/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-indigo-900/10 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-4xl mx-auto relative z-10">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in">
          <div className="inline-flex items-center justify-center p-4 mb-6 bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-full shadow-lg ring-1 ring-slate-700/50">
            <Trophy className="h-8 w-8 text-yellow-500" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
            Interview Complete
          </h1>
          <p className="text-xl text-slate-400">
            Here's a detailed breakdown of your performance
          </p>
        </div>

        {/* Overall Score */}
        <Card className="p-8 mb-8 bg-slate-900/60 backdrop-blur-xl border-slate-800 shadow-2xl rounded-3xl animate-slide-up overflow-hidden relative">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500"></div>
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="text-center md:text-left">
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-1">Overall Score</h2>
              <div className="flex items-baseline gap-2 justify-center md:justify-start">
                <span className="text-7xl font-bold text-white tracking-tighter">
                  {overallScore}
                </span>
                <span className="text-2xl text-slate-500 font-medium">%</span>
              </div>
              <div className="mt-4 flex flex-wrap gap-2 justify-center md:justify-start">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  <Target className="w-3 h-3 mr-1" />
                  {answeredCount}/{totalQuestions} Answered
                </span>
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  <Trophy className="w-3 h-3 mr-1" />
                  Top 15%
                </span>
              </div>
            </div>

            <div className="flex-1 w-full md:max-w-md bg-slate-950/50 rounded-2xl p-6 border border-slate-800/50">
              <h3 className="text-slate-300 font-medium mb-3 flex items-center gap-2">
                <Lightbulb className="w-4 h-4 text-yellow-500" />
                AI Feedback Summary
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                {session.result_summary || "Great job! You demonstrated strong potential. Focus on providing more specific examples and elaborate on technical constraints to improve further."}
              </p>
            </div>
          </div>
        </Card>

        {/* Score Breakdown */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card className="p-6 bg-slate-900/40 backdrop-blur-lg border-slate-800 hover:border-blue-500/30 transition-all duration-300 rounded-2xl group">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-blue-500/10 text-blue-400 group-hover:bg-blue-500 group-hover:text-white transition-colors duration-300">
                  <Target className="h-5 w-5" />
                </div>
                <h3 className="font-semibold text-slate-200">Communication</h3>
              </div>
              <span className="text-2xl font-bold text-white">{communicationScore}%</span>
            </div>
            <Progress value={communicationScore} className="h-2 bg-slate-800" indicatorClassName="bg-blue-500" />
            <p className="text-sm text-slate-500 mt-4 leading-relaxed">
              Measures clarity, articulation, and pacing of your responses.
            </p>
          </Card>

          <Card className="p-6 bg-slate-900/40 backdrop-blur-lg border-slate-800 hover:border-purple-500/30 transition-all duration-300 rounded-2xl group" style={{ animationDelay: "0.1s" }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400 group-hover:bg-purple-500 group-hover:text-white transition-colors duration-300">
                  <Lightbulb className="h-5 w-5" />
                </div>
                <h3 className="font-semibold text-slate-200">Technical Skills</h3>
              </div>
              <span className="text-2xl font-bold text-white">{technicalScore}%</span>
            </div>
            <Progress value={technicalScore} className="h-2 bg-slate-800" indicatorClassName="bg-purple-500" />
            <p className="text-sm text-slate-500 mt-4 leading-relaxed">
              Evaluates accuracy, depth of knowledge, and problem-solving approach.
            </p>
          </Card>
        </div>

        {/* Detailed Answers */}
        {session.answers && session.answers.length > 0 && (
          <div className="mb-12">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <span className="w-1 h-6 bg-blue-500 rounded-full"></span>
              Detailed Analysis
            </h3>

            <div className="space-y-4">
              {session.answers.map((answer, index) => {
                const accuracyScore = answer.accuracy_score !== null && answer.accuracy_score !== undefined
                  ? Math.round(answer.accuracy_score * 100)
                  : answer.topic_score !== null
                    ? Math.round(answer.topic_score * 100)
                    : Math.round(answer.similarity_score * 100);

                return (
                  <Card key={answer.id} className="overflow-hidden bg-slate-900/40 border-slate-800 hover:bg-slate-900/60 transition-colors duration-300 rounded-2xl group">
                    <div className="p-6">
                      <div className="flex items-start gap-4">
                        <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-slate-800 text-slate-400 font-mono text-sm border border-slate-700">
                          {index + 1}
                        </span>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-slate-200 mb-2 leading-relaxed">
                            {answer.question_text}
                          </h4>
                          <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-800/50 mb-4">
                            <p className="text-sm text-slate-400 italic">
                              "{answer.user_answer}"
                            </p>
                          </div>

                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-800/50">
                            <div>
                              <div className="text-xs text-slate-500 mb-1">Accuracy</div>
                              <div className="text-lg font-bold text-blue-400">{accuracyScore}%</div>
                            </div>
                            <div>
                              <div className="text-xs text-slate-500 mb-1">Concept Match</div>
                              <div className="text-lg font-bold text-purple-400">{Math.round(answer.similarity_score * 100)}%</div>
                            </div>
                            {answer.matched_keywords_list && (
                              <div className="col-span-2">
                                <div className="text-xs text-slate-500 mb-2">Keywords Detected</div>
                                <div className="flex flex-wrap gap-1.5">
                                  {answer.matched_keywords_list.slice(0, 4).map((kw, i) => (
                                    <span key={i} className="px-2 py-0.5 rounded text-[10px] uppercase font-medium bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                                      {kw}
                                    </span>
                                  ))}
                                  {answer.matched_keywords_list.length > 4 && (
                                    <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-500">
                                      +{answer.matched_keywords_list.length - 4}
                                    </span>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          </div>
        )}

        {/* Improvements */}
        <Card className="p-8 mb-12 bg-gradient-to-br from-slate-900 to-slate-900/50 border-slate-800 rounded-3xl">
          <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Lightbulb className="h-6 w-6 text-yellow-500" />
            Recommended Improvements
          </h3>
          {improvements.length > 0 ? (
            <div className="grid sm:grid-cols-2 gap-4">
              {improvements.map((improvement, index) => (
                <div key={index} className="flex items-start gap-3 p-4 bg-slate-950/50 rounded-xl border border-slate-800 hover:border-slate-700 transition-colors">
                  <span className="text-blue-500 mt-0.5">•</span>
                  <p className="text-slate-300 text-sm leading-relaxed">{improvement}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-slate-400 italic">No specific improvements identified. Excellent performance!</p>
          )}
        </Card>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center pb-12">
          <Button
            onClick={() => navigate("/")}
            variant="outline"
            className="px-8 py-6 text-lg bg-transparent border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white rounded-full transition-all"
            size="lg"
          >
            <Home className="mr-2 h-5 w-5" />
            Back to Home
          </Button>
          <Button
            onClick={handleDownload}
            className="px-8 py-6 text-lg bg-white text-slate-950 hover:bg-slate-200 rounded-full font-semibold shadow-[0_0_20px_-5px_rgba(255,255,255,0.3)] transition-all hover:scale-105"
            size="lg"
          >
            <Download className="mr-2 h-5 w-5" />
            Download Summary
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Results;
