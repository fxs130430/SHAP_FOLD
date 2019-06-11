import java.util.ArrayList;
import java.util.regex.*;
import javafx.util.Pair;
import org.jpl7.Query;

public class Clause {
    
    public Predicate Head;
    public ArrayList<Predicate> Body;
    public boolean bMarkAsDeleted;
    
    public Clause(String x_strClause)
    {
        Body = new ArrayList<>();
        bMarkAsDeleted = false;
        if(x_strClause.contains(":-"))
        { 
            String[] arrSubs = x_strClause.split(":-");
            Head = new Predicate(arrSubs[0]);
            if(!arrSubs[1].contains("true"))
            {
                String regex = "\\w+\\([a-zA-Z0-9\\.,_]+\\)|not\\(\\w+\\([a-zA-Z0-9\\.,_]+\\)\\)|[a-zA-Z0-9]+\\s*[!|=]=\\s*[a-zA-Z0-9]+|V[0-9]+\\s*[>|<]=\\s*[0-9]+";
                Pattern pattern = Pattern.compile(regex, Pattern.MULTILINE);
                Matcher matcher = pattern.matcher(arrSubs[1]);
                while (matcher.find()) {
                    Body.add(new Predicate(matcher.group(0)));
                }
            }
        }
        else // It is a fact
        {
            Head = new Predicate(x_strClause);
        }
    }
    public String[] getBodyPredicates()
    {
        String[] ret_arr = new String[this.Body.size()];
        
        for(int i = 0 ; i < this.Body.size() ; i++)
            ret_arr[i] = Body.get(i).toString();
        return ret_arr;
    }
    public int getPredicateCount(String x_strPredName)
    {
        int nCount = 0;
        for(Predicate b:Body)
            if(b.name.equals(x_strPredName))
                nCount++;
        return nCount;
    }
    public void addPredicate(Predicate p)
    {
        Body.add(p);
    }
    public boolean IsFact()
    {
        return Body.isEmpty();
    }
    public ArrayList<Clause> remove_predicates_one_at_a_time()
    {
        ArrayList<Clause> arrTemp = new ArrayList<>();
        if(this.Body.size() == 1)
            return arrTemp;
        for(int i = 0 ; i < Body.size() ; i++)
        {
            Clause c_new = new Clause(this.toString());
            c_new.Body.remove(i);
            arrTemp.add(c_new);            
        }
        return arrTemp;
    }
    public ArrayList<String> getVars()
    {
        ArrayList<String> arrVars = new ArrayList<>();
        for(Pair p:Head.getVars_withIndex())
            arrVars.add(p.getValue().toString());
        for(Predicate b:Body)
            for(Pair p:b.getVars_withIndex())
                arrVars.add(p.getValue().toString());
        return arrVars;
    } 
    public boolean retract_clause()
    {
        boolean bRet = false;
        String t = "retractall(("+Head.toString()+"))";
        Query q = new Query(t);
        if(q.hasSolution())
            bRet = true;
        return bRet;
    }
    public boolean assert_clause()
    {
        boolean bRet = false;
        String t  = "assert(("+toString()+"))";
        Query q = new Query(t);
        if(q.hasSolution())
            bRet = true;    
        return bRet;
    }
    @Override
    public String toString()
    {
        StringBuilder buff = new StringBuilder();
        buff.append(Head.toString());
        if(Body.isEmpty())
           buff.append(":- true");
        else
        {    
            buff.append(":-");
            for(int i = 0 ; i < Body.size() ; i++)
            {
                buff.append(Body.get(i).toString());
                if(i < Body.size() - 1)
                    buff.append(",");

            }
        }
        return buff.toString();       
    }
}
