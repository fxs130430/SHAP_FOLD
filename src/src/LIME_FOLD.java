import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javafx.util.Pair;
import org.jpl7.Compound;
import org.jpl7.Query;
import org.jpl7.Term;
import org.jpl7.Util;

public class LIME_FOLD 
{
    private Query                          m_qfoil;
    private Query                          m_qDS;
    private LanguageBias                   m_Language_Bias;
    public ArrayList<Predicate>            m_posExamples;
    public ArrayList<Predicate>            m_negExamples;
    private ArrayList<Clause>              m_Clauses;
    private ArrayList<Clause>              m_ExceptionClauses;
    private int                            m_nAbIndex;
    private Map<String,Map<String,Double>> m_mapSHAPVals;
   
    
    public LIME_FOLD(String x_strDataset,String x_strModeDeclarations,String x_strSHAP_URL)
    {
        m_Language_Bias = new LanguageBias(x_strModeDeclarations);
        String t1 = "consult('foil.pl')";
        String t2 = "consult('"+x_strDataset+"')";
        m_qfoil = new Query(t1);
        if(!m_qfoil.hasSolution())
        {
            System.out.println("failed to initialized foil...");
            System.exit(-1);
        }
        else
        {
            m_qDS = new Query(t2);
            if(m_qDS.hasSolution())
            {
                System.out.printf("%s initialized successfully...\r\n","foil.pl");
                System.out.printf("%s initialized successfully...\r\n",x_strDataset);
            }
            else
            {
                System.out.printf("failed to load dataset %s...\r\n",x_strDataset);
                System.exit(-1);
            }
        }
        m_posExamples = new ArrayList<>();
        m_negExamples = new ArrayList<>();
        
        m_Clauses = new ArrayList<>();
        m_ExceptionClauses = new ArrayList<>();
        
        if(!getExamples())
        {
                System.out.printf("failed to load examples...\r\n");
                System.exit(-1);
        }
        m_mapSHAPVals = new HashMap<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(x_strSHAP_URL))) 
        {
            String line;
            String[] header = null;
            while ((line = br.readLine()) != null) 
            {
                header = line.split(":");
                if(!m_mapSHAPVals.containsKey(header[0]))
                    m_mapSHAPVals.put(header[0], new HashMap<>());
                m_mapSHAPVals.get(header[0]).put(header[1], Double.parseDouble(header[2]));
            }
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
            System.exit(-1);
        }
    }   
    private boolean getExamples()
    {
        Clause cl_goal = m_Language_Bias.getGoalClause();
        String t1 = "get_examples("+cl_goal.Head.toString()+",Positives,Negatives)";
        Query q = new Query(t1);
        if(q.hasSolution())
        {
            Compound comp_pos = (Compound)q.oneSolution().get("Positives");
            Compound comp_neg = (Compound)q.oneSolution().get("Negatives");
            
            for(Term t:comp_pos.toTermArray())
            {
                //System.out.println(t.toString());
                m_posExamples.add(new Predicate(t.toString()));
            }
            for(Term t:comp_neg.toTermArray())
            {
                if(t.toString().equals("approved(p)"))
                    continue;
                if(t.toString().equals("ckd(poor)"))
                    continue;
                if(!t.toString().contains("(p"))
                    continue;
                //System.out.println(t.toString());
                m_negExamples.add(new Predicate(t.toString()));
            }
            return true;
        }
        return false;        
    }
    public ArrayList<Predicate> covered_examples(Clause x_clause,ArrayList<Predicate> x_predicates)
    {
        ArrayList<Predicate> arrTemp = new ArrayList<>();
        String strPos_list = utils.toPrologList(x_predicates);
        String t1 = "covered_examples(("+x_clause.toString()+"), "+strPos_list+", Xs1)";
        Query q = new Query(t1);
        if(q.hasSolution())
        {
            Term t = (Term)q.oneSolution().get("Xs1");
            Term[] sub_t = Util.listToTermArray(t);
            for(Term tt: sub_t)
            {
                arrTemp.add(new Predicate(tt.toString()));
                //System.out.println(tt.toString());
            }
        }
        else
        {
            System.out.println("Error in covered_examples");
            System.exit(-1);
        } 
        return arrTemp;
    }
    public ArrayList<Predicate> update_covered_examples(Clause x_clause,ArrayList<Predicate> x_PosTerms)
    {
        ArrayList<Predicate> arrUncovered = new ArrayList<>();
        String strPos_list = utils.toPrologList(x_PosTerms);
        String t1 = "uncovered_examples(("+x_clause.toString()+"), "+strPos_list+", Xs1)";
        Query q = new Query(t1);
        if(q.hasSolution())
        {
            Term t = (Term)q.oneSolution().get("Xs1");
            Term[] sub_t = Util.listToTermArray(t);
            for(Term tt: sub_t)
            {
                arrUncovered.add(new Predicate(tt.toString()));
                //System.out.println(tt.toString());
            }            
        }
        else
        {
            System.out.println("Error in update_covered_examples");
            System.exit(-1);
        }        
        return arrUncovered;
    }
    private double getInfo_value(Clause x_clause,ArrayList<Predicate> x_PosTerms,ArrayList<Predicate> x_NegTerms)
    {
        String strPos_list = utils.toPrologList(x_PosTerms);
        String strNeg_list = utils.toPrologList(x_NegTerms);
        String t1 = "info_value(("+x_clause.toString()+"), "+strPos_list+","+strNeg_list+", Value)";
        Query q1 = new Query(t1);
        if(q1.hasSolution())
        {
            double dGain = ((Term)q1.oneSolution().get("Value")).doubleValue();
            return dGain;
        }
        System.out.println("Error in getInfo_value");
        System.exit(-1);
        return 0;        
    }
    public double compute_gain(Clause x_clause,double x_dCurrentInfo,ArrayList<Predicate> x_PosPredicates,ArrayList<Predicate> x_NegPredicates)
    {
        String strPos_list = utils.toPrologList(x_PosPredicates);
        String strNeg_list = utils.toPrologList(x_NegPredicates);
        String t1 = "compute_gain("+strNeg_list+", "+strPos_list+", "+x_dCurrentInfo+", ("+x_clause.toString()+"),Gain)";
        Query q1 = new Query(t1);
        if(q1.hasSolution())
        {
            double dGain = ((Term)q1.oneSolution().get("Gain")).doubleValue();
            return dGain;
        }
        System.out.println("Error in gain computation");
        System.exit(-1);
        return 0;
    }
    public double getRuleAccuracy(Clause x_clause,ArrayList<Predicate> x_posExamples, ArrayList<Predicate> x_negExamples)
    {
        double tp = covered_examples(x_clause, x_posExamples).size();
        double tn = x_negExamples.size() - covered_examples(x_clause, x_negExamples).size();
        double total = x_posExamples.size() + x_negExamples.size();
        return (tp + tn) / total;
    }
    
    private double getExplicitBits(Clause x_cl,ArrayList<Predicate> x_arrCurrentExamples)
    {
        /* |T| -> Cardinality of the training set (E+ and E-)
        /* explicitbits(cl) = log2(|T|) + log2(choose(|T|, covered(cl,E+))
        */
        //int T = x_arrCurrentExamples.size();
        int T = m_posExamples.size() + m_negExamples.size();
        int n_plus = covered_examples(x_cl, x_arrCurrentExamples).size();
        if(n_plus == 0) // To avoid Log(0) 
            return 0;
        double f1 = Math.log(T) / Math.log(2);
        double choose = utils.choose_n_k(T, n_plus);
        double f2 = Math.log(choose) / Math.log(2);
        return f1 + f2;
    }
    private double getClauseBits(Clause x_cl)
    {
        double dTotal =0;
        for(Predicate p:x_cl.Body)
        {
            double l = m_Language_Bias.getDifferentLiteralsBits();
            double variablization = Math.pow(2,p.getVarNames().size()) - 1;
            dTotal = dTotal  + l + 1;
        }        
        dTotal -= Math.log(utils.factorial(x_cl.Body.size())) / Math.log(2);
        return dTotal;
    }
    public ArrayList<String> best_next_clause_SHAP(Clause x_clause,ArrayList<Predicate> x_PosExamples,ArrayList<Predicate> x_NegExamples)
    {
        Map<String, Double> mapSumAllShapScores = new HashMap<String, Double>();
        for(Predicate e: x_PosExamples)
        {
            double sign = 0;
            if(e.toString().contains("positive"))
            {    sign = 1;
                if(m_negExamples.contains(e))
                    sign = -1;
            }
            else if(e.toString().contains("negative"))
            {
                sign = -1;
                if(m_negExamples.contains(e))
                    sign = 1;
            }
            String id = e.toString().substring(e.toString().indexOf("(")+ 1, e.toString().indexOf(")"));
            String shap_key = String.format("data(%s)",id); 
                
            if(m_mapSHAPVals.containsKey(shap_key))
            {
                Map<String, Double> map_e = m_mapSHAPVals.get(shap_key);
                for (Map.Entry<String, Double> entry : map_e.entrySet())
                {
                    if(!mapSumAllShapScores.containsKey(entry.getKey()))
                        mapSumAllShapScores.put(entry.getKey(),0.0);

                    double temp = (sign * entry.getValue()) + mapSumAllShapScores.get(entry.getKey());
                    mapSumAllShapScores.put(entry.getKey(),temp);
                }
            }
            else
            {
                System.out.println("What the F?!");
            }
        }
        
        mapSumAllShapScores = utils.sortByValue((HashMap<String, Double>)mapSumAllShapScores);
        ArrayList<String> arrTop_SHAP = new ArrayList<>();
        for(String s:mapSumAllShapScores.keySet())
        {
            arrTop_SHAP.add(s);
            if(arrTop_SHAP.size() > 5)
                break;
        }
        return arrTop_SHAP;
    }
    public Pair<Clause,Double> best_next_clause(Clause x_clause,ArrayList<Predicate> x_PosExamples,ArrayList<Predicate> x_NegExamples)
    {
        double dCurrent_Info = getInfo_value(x_clause, x_PosExamples, x_NegExamples);
        Clause c_best = null;
        double dGain_best = 0;
        ArrayList<Clause> arrCandidate_clauses = m_Language_Bias.refine(x_clause);
        int total = arrCandidate_clauses.size();
        int n = 0;
        
        ArrayList<String> top_shap_preds = best_next_clause_SHAP(x_clause, x_PosExamples, x_NegExamples);
        for(Clause c:arrCandidate_clauses)
        {
            //System.out.println(String.format("%d of %d",++n,total));
            double dGain = compute_gain(c, dCurrent_Info, x_PosExamples, x_NegExamples);
            //ArrayList<Predicate> covered_examples = covered_examples(c, x_PosExamples);
            //ArrayList<String> top_shap_preds = best_next_clause_SHAP(c, covered_examples, x_NegExamples);

            if(dGain > dGain_best)
            {
                String[] body_preds = c.getBodyPredicates(); 
                String strLastPred = body_preds[body_preds.length - 1];
                if(top_shap_preds.contains(strLastPred))
                {
                    dGain_best = dGain;
                    c_best = c;
                }
                else
                {
                    System.out.println("shap discretion!");
                }
            }
        }
        if(c_best == null)
            return null;
        double dExplictBits = getExplicitBits(c_best,x_PosExamples);
        double dHypothesisEncoding = getClauseBits(c_best);
        double dMDL_score = dExplictBits - dHypothesisEncoding;
        if(dMDL_score < 0)
            return null;
        return new Pair<Clause,Double>(c_best,dGain_best);       
    }
    private Clause extend_clause_loop(Clause x_clause,ArrayList<Predicate> x_PosExamples,ArrayList<Predicate> x_NegExamples)
    {
        Clause c_new = null;
        boolean bExit = false;
        
        while(!bExit)
        {
            //Pair<Clause,Double> pair = best_next_clause_SHAP(x_clause,x_PosExamples,x_NegExamples);
            Pair<Clause,Double> pair = best_next_clause(x_clause,x_PosExamples,x_NegExamples);
            
            if(pair != null) 
            {
                 c_new = pair.getKey();
                 x_clause = c_new;
            }
            else
            {
                Pair<Clause,ArrayList<Clause>> pair_exception = exception_handler(x_clause, x_NegExamples, x_PosExamples);
                if(pair_exception != null)
                {
                    c_new = pair_exception.getKey();
                    for(Clause c: pair_exception.getValue())
                        m_ExceptionClauses.add(c);                    
                }
                //If the current hypothesis covers at least 85 % of the examples, keep it.
                bExit = true;
                if(x_clause.Body.size() > 0)
                {
                    if(getRuleAccuracy(x_clause, x_PosExamples, x_NegExamples) < 0.8)
                        c_new = null;
                    else
                        c_new = x_clause;
                }
                else c_new = null;
            }
            if(c_new == null)
            {
                bExit = true;
            }
            else
            {
                x_PosExamples = covered_examples(c_new, x_PosExamples);
                x_NegExamples = covered_examples(c_new, x_NegExamples);
                
                if(getRuleAccuracy(c_new, x_PosExamples, x_NegExamples) > 0.8)
                //if(x_NegExamples.size() == 0)
                    bExit = true;
            }
        }
        return c_new;
    }
    private void foil_loop(ArrayList<Predicate> x_PosExamples,ArrayList<Predicate> x_NegExamples,ArrayList<Clause> x_Clauses)
    {
        boolean bExit = false;
        int nFailCounter = 0;
        while(!bExit)
        {
            Clause MostGenralClause = m_Language_Bias.getGoalClause();
            Clause c_new = extend_clause_loop(MostGenralClause,x_PosExamples,x_NegExamples);
            if(c_new == null)
            {
                System.out.println("Failed to cover anything...");
                nFailCounter++;
            }
            else
            {
                if(c_new.Body.size() > 0) //not the Most Genral Clause
                {
                    nFailCounter = 0;
                    x_PosExamples = update_covered_examples(c_new,x_PosExamples);
                    x_Clauses.add(c_new);
                    System.out.printf("Clause added:>>>> %s\r\n", c_new);
                }
            }
            if(nFailCounter == 2 || x_PosExamples.size() == 0)
                bExit = true;            
        }        
    }
    private Pair<Clause,ArrayList<Clause>> exception_handler(Clause x_curClause,ArrayList<Predicate> x_PosExamples,ArrayList<Predicate> x_NegExamples)
    {
        ArrayList<Clause> arrAbPredicates = new ArrayList<>();
        Pair<Clause,Double> p = best_next_clause(x_curClause, x_PosExamples, x_NegExamples);
        //Pair<Clause,Double> p = best_next_clause_SHAP(x_curClause, x_PosExamples, x_NegExamples);
        Clause c_ret = null;
        
        if(p != null) // positive gain
        {
            ArrayList<Clause> arrClauses = new ArrayList<>();
            foil_loop(x_PosExamples, x_NegExamples,arrClauses);
            if(arrClauses.size() > 0)
            {
                String strHead = String.format("ab%d", m_nAbIndex++);
                for(int i = 0 ; i < arrClauses.size() ; i++)
                {
                    Clause c = arrClauses.get(i);
                    c.Head.name = strHead;
                    arrAbPredicates.add(c);
                    if(c.assert_clause())
                    {
                        System.out.printf("%s added to the exceptions.\r\n", c.toString());
                    }
                    else
                    {
                        System.out.printf("Error in asserting clause %s\r\n",c.toString());
                        System.exit(-1);
                    }
                }
                Predicate negated_p = new Predicate(String.format("not(%s)", arrClauses.get(0).Head.toString()));
                x_curClause.addPredicate(negated_p);
                c_ret = x_curClause;
            }
        }
        return new Pair<Clause,ArrayList<Clause>>(c_ret,arrAbPredicates);
    }
    public void foil_run()
    {
        foil_loop(m_posExamples, m_negExamples, m_Clauses);
        System.out.println("Learned concept clause(s):");
        for(Clause c: m_Clauses)
        {
            System.out.printf("%s.\r\n", c.toString());
        }
        System.out.println("Exception clause(s):");
        for(Clause c: m_ExceptionClauses)
        {
            System.out.printf("%s.\r\n", c.toString());
        }
    } 
    private int getTotalCoverage(ArrayList<Clause> x_arrClauses)
    {
        Set<String> setCovered = new HashSet<String>();
        for(Clause c: x_arrClauses)
        {
            if(c.bMarkAsDeleted)
                continue;
            ArrayList<Predicate> arrTemp = covered_examples(c, m_posExamples);
            for(Predicate p: arrTemp)
                setCovered.add(p.toString());
        }        
        return setCovered.size();
    }
    public void post_processing()
    {
        ArrayList<Clause> m_ppClauses = new ArrayList<>();
        HashMap<String,Pair<Integer,Integer>> hCoverage = new HashMap<>();
        for(Clause c:m_Clauses)
        {
            int p = covered_examples(c, m_posExamples).size();
            int n = covered_examples(c, m_negExamples).size();
            //hCoverage.put(c.toString(), new Pair<Integer,Integer>(p,n));
            ArrayList<Clause> arrAlternatives = c.remove_predicates_one_at_a_time();
            
            Clause c_new = null;
            for(Clause cc:arrAlternatives)
            {
                int p_new = covered_examples(cc, m_posExamples).size();
                int n_new = covered_examples(cc, m_negExamples).size();
                if(p_new >= p && n_new <= n)
                {
                    c_new = cc;
                    break;
                }
            }
            if(c_new != null)
            {
                m_ppClauses.add(c_new);
                System.out.printf("old: %s\r\n",c.toString());
                System.out.printf("new: %s\r\n",c_new.toString());
            }
            else
            {
                m_ppClauses.add(c);
            }
        }
        int nOriginal_Coverage = getTotalCoverage(m_ppClauses);
        ArrayList<Pair<Integer,Clause>> arrCoverage = new ArrayList<>();
        for(Clause c:m_ppClauses)
        {
            int pos = covered_examples(c, m_posExamples).size();
            arrCoverage.add(new Pair<Integer,Clause>(pos,c));
        }
        Collections.sort(arrCoverage, new Comparator<Pair<Integer,Clause>>() 
        {
            @Override
            public int compare(Pair<Integer,Clause> p1, Pair<Integer,Clause> p2) {
            return p1.getKey().compareTo(p2.getKey());}
        });
        for(Pair<Integer,Clause> p: arrCoverage)
        {
            p.getValue().bMarkAsDeleted = true;
            if((double)(nOriginal_Coverage - getTotalCoverage(m_ppClauses)) / (double)nOriginal_Coverage > 0.03 )
                p.getValue().bMarkAsDeleted = false;
            else
                System.out.printf("Rule [ %s ]  is redundant!\r\n", p.getValue().toString());
        }
        m_Clauses = m_ppClauses;
    }
    public void Display_Hypothesis()
    {
        for(Clause c: m_Clauses)
        {
            if(!c.bMarkAsDeleted)
                System.out.printf("%s.\r\n",c.toString());
        }
        System.out.println();
        for(Clause c:m_ExceptionClauses)
            System.out.printf("%s.\r\n",c.toString());
        
        System.out.printf("Total Coverage: %d out of %d\r\n", getTotalCoverage(m_Clauses),m_posExamples.size());
    }
    
}
