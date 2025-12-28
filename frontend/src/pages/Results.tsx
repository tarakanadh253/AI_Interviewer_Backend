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
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <Loader2 className="h-10 w-10 animate-spin mx-auto mb-4 text-ohg-orange" />
          <p className="text-muted-foreground">Analyzing performance...</p>
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-6">
        <Card className="p-10 max-w-md w-full bg-white border-border text-center shadow-2xl rounded-3xl">
          <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-6">
            <Trophy className="h-8 w-8 text-muted-foreground" />
          </div>
          <h2 className="text-2xl font-bold text-ohg-navy mb-2">No Results Found</h2>
          <p className="text-muted-foreground mb-8">
            Complete an interview to see your detailed analysis here.
          </p>
          <Button
            onClick={() => navigate("/")}
            className="w-full bg-ohg-navy text-white hover:bg-ohg-navy/90 h-12 rounded-xl font-bold"
          >
            Return Home
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground p-6 relative overflow-hidden">
      {/* Background Ambience */}
      <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-ohg-navy/5 rounded-full blur-[100px] pointer-events-none -translate-y-1/2 translate-x-1/2" />
      <div className="absolute bottom-0 left-0 w-[800px] h-[800px] bg-ohg-orange/5 rounded-full blur-[100px] pointer-events-none translate-y-1/2 -translate-x-1/2" />

      <div className="max-w-4xl mx-auto relative z-10">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in">
          <div className="inline-flex items-center justify-center p-4 mb-6 bg-card border border-border rounded-full shadow-lg ring-1 ring-black/5">
            <Trophy className="h-8 w-8 text-ohg-orange" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-ohg-navy dark:text-foreground mb-4 tracking-tight">
            Interview Complete
          </h1>
          <p className="text-xl text-muted-foreground">
            Here's a detailed breakdown of your performance
          </p>
        </div>

        {/* Overall Score */}
        <Card className="p-8 mb-8 bg-card/80 backdrop-blur-xl border-border shadow-soft rounded-3xl animate-slide-up overflow-hidden relative">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-ohg-orange via-ohg-teal to-ohg-navy"></div>
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="text-center md:text-left">
              <h2 className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-1">Overall Score</h2>
              <div className="flex items-baseline gap-2 justify-center md:justify-start">
                <span className="text-7xl font-bold text-foreground tracking-tighter">
                  {overallScore}
                </span>
                <span className="text-2xl text-muted-foreground font-medium">%</span>
              </div>
              <div className="mt-4 flex flex-wrap gap-2 justify-center md:justify-start">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700 border border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-400 dark:border-emerald-800">
                  <Target className="w-3 h-3 mr-1" />
                  {answeredCount}/{totalQuestions} Answered
                </span>
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-ohg-teal/10 text-ohg-teal border border-ohg-teal/20">
                  <Trophy className="w-3 h-3 mr-1" />
                  Top 15%
                </span>
              </div>
            </div>

            <div className="flex-1 w-full md:max-w-md bg-muted/50 rounded-2xl p-6 border border-border">
              <h3 className="text-foreground font-bold mb-3 flex items-center gap-2">
                <Lightbulb className="w-4 h-4 text-ohg-orange" />
                AI Feedback Summary
              </h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                {session.result_summary || "Great job! You demonstrated strong potential. Focus on providing more specific examples and elaborate on technical constraints to improve further."}
              </p>
            </div>
          </div>
        </Card>

        {/* Score Breakdown */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card className="p-6 bg-card/60 backdrop-blur-lg border-border hover:border-ohg-teal/30 transition-all duration-300 rounded-2xl group shadow-sm hover:-translate-y-1">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-ohg-teal/10 text-ohg-teal group-hover:bg-ohg-teal group-hover:text-white transition-colors duration-300">
                  <Target className="h-5 w-5" />
                </div>
                <h3 className="font-bold text-foreground">Communication</h3>
              </div>
              <span className="text-2xl font-bold text-foreground">{communicationScore}%</span>
            </div>
            <Progress value={communicationScore} className="h-2 bg-muted" indicatorClassName="bg-ohg-teal" />
            <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
              Measures clarity, articulation, and pacing of your responses.
            </p>
          </Card>

          <Card className="p-6 bg-card/60 backdrop-blur-lg border-border hover:border-ohg-navy/30 transition-all duration-300 rounded-2xl group shadow-sm hover:-translate-y-1" style={{ animationDelay: "0.1s" }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-ohg-navy/10 text-ohg-navy dark:text-foreground group-hover:bg-ohg-navy group-hover:text-white transition-colors duration-300">
                  <Lightbulb className="h-5 w-5" />
                </div>
                <h3 className="font-bold text-foreground">Technical Skills</h3>
              </div>
              <span className="text-2xl font-bold text-foreground">{technicalScore}%</span>
            </div>
            <Progress value={technicalScore} className="h-2 bg-muted" indicatorClassName="bg-ohg-navy" />
            <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
              Evaluates accuracy, depth of knowledge, and problem-solving approach.
            </p>
          </Card>
        </div>

        {/* Detailed Answers */}
        {session.answers && (
          <div className="mb-12">
            <h3 className="text-xl font-bold text-foreground mb-6 flex items-center gap-2">
              <span className="w-1 h-6 bg-ohg-orange rounded-full"></span>
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
                  <Card key={answer.id} className="overflow-hidden bg-white border border-gray-100 hover:border-ohg-teal/30 transition-all duration-300 rounded-2xl shadow-sm relative group">
                    <div className="p-6 md:p-8">
                      <div className="flex flex-col md:flex-row gap-6">
                        {/* Number Badge */}
                        <div className="flex-shrink-0 pt-1">
                          <div className="w-8 h-8 flex items-center justify-center rounded-lg bg-ohg-navy/5 text-ohg-navy font-mono text-sm font-bold border border-ohg-navy/10">
                            {index + 1}
                          </div>
                        </div>

                        {/* Content */}
                        <div className="flex-1 space-y-6">
                          {/* Question */}
                          <h4 className="text-lg font-bold text-gray-900 leading-relaxed tracking-tight">
                            {answer.question_text}
                          </h4>

                          {/* Answer Box */}
                          <div className="bg-gray-50 p-6 rounded-xl border border-gray-100 relative overflow-hidden">
                            <div className="relative z-10">
                              <p className="text-gray-600 text-sm leading-7 italic font-medium">
                                "{answer.user_answer}"
                              </p>
                            </div>
                          </div>

                          {/* Metrics Footer */}
                          <div className="flex flex-wrap items-center gap-8 md:gap-16 pt-2 border-t border-gray-100">
                            <div>
                              <div className="text-xs text-gray-400 font-medium uppercase tracking-wider mb-1">Marks Obtained</div>
                              <div className="text-2xl font-bold text-ohg-teal">{Math.round((accuracyScore / 100) * 10)}/10</div>
                            </div>
                            <div>
                              <div className="text-xs text-gray-400 font-medium uppercase tracking-wider mb-1">Topic Match</div>
                              <div className="text-2xl font-bold text-ohg-navy">{Math.round(answer.similarity_score * 10)}/10</div>
                            </div>
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
        <Card className="p-8 mb-12 bg-card shadow-soft border-border rounded-3xl">
          <h3 className="text-xl font-bold text-foreground mb-6 flex items-center gap-2">
            <Lightbulb className="h-6 w-6 text-ohg-orange" />
            Recommended Improvements
          </h3>
          {improvements.length > 0 ? (
            <div className="grid sm:grid-cols-2 gap-4">
              {improvements.map((improvement, index) => (
                <div key={index} className="flex items-start gap-3 p-4 bg-muted/30 rounded-xl border border-border hover:border-ohg-teal/30 transition-colors">
                  <span className="text-ohg-teal mt-0.5">•</span>
                  <p className="text-muted-foreground text-sm leading-relaxed font-medium">{improvement}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground italic">No specific improvements identified. Excellent performance!</p>
          )}
        </Card>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center pb-12">
          <Button
            onClick={() => navigate("/")}
            variant="outline"
            className="px-8 py-6 text-lg bg-card border-border text-foreground hover:bg-muted hover:text-foreground rounded-full transition-all font-medium"
            size="lg"
          >
            <Home className="mr-2 h-5 w-5" />
            Back to Home
          </Button>
          <Button
            onClick={handleDownload}
            className="px-8 py-6 text-lg bg-ohg-orange text-white hover:bg-ohg-orange-hover rounded-full font-bold shadow-xl hover:shadow-ohg-orange/30 transition-all hover:scale-105"
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
