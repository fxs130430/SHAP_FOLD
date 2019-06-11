import java.util.ArrayList;
//import org.jpl7.*;
import javafx.util.Pair;

public class Predicate {
    public String   name;
    public int      arity;
    public String   sign;
    public String[] args;
    
    public Predicate(String x_strPred)
    {
        this.sign = "+";
        if( x_strPred.startsWith("not("))
        {
            this.sign = "-";
            x_strPred = x_strPred.substring(x_strPred.indexOf("(") + 1,x_strPred.lastIndexOf(")"));
        }
        else if(x_strPred.startsWith("\\+"))
        {
            this.sign = "-";
            x_strPred = x_strPred.substring(x_strPred.indexOf("+")+1).trim();
        }
        this.name = x_strPred.substring(0,x_strPred.indexOf("("));
        String strArgs_list = x_strPred.substring(x_strPred.indexOf("(") + 1,x_strPred.indexOf(")"));
        this.args = strArgs_list.split(",");  
        this.arity = args.length;
    }
    public Predicate Copy()
    {
        return new Predicate(this.toString());
    }
    public ArrayList<Pair<Integer,String>> getVars_withIndex()
    {
        ArrayList<Pair<Integer,String>> var_list = new ArrayList();
        for(int i = 0 ; i < args.length ; i++)
            if(Character.isUpperCase(args[i].charAt(0))) // It's a variable
                var_list.add(new Pair<Integer,String>(i,args[i]));
        return var_list;        
    }
    public ArrayList<String> getVarNames()
    {
        ArrayList<String> var_list = new ArrayList();
        for(int i = 0 ; i < args.length ; i++)
            if(Character.isUpperCase(args[i].charAt(0))) // It's a variable
                var_list.add(args[i]);
        return var_list;  
    }
    /*
    public Term ToJPLTerm()
    {
        Compound c = new Compound(name, new Term[arity]);
        for(int i = 0 ; i < arity ; i++)
        {
            Term t;
            if(Character.isUpperCase(args[i].charAt(0))) // It's a variable
                t = new Variable(args[i]);
            else
                t = new Atom(args[i]);
            c.args()[i] = t;
        }
        return c;
    }
   */
    @Override
    public String toString()
    {
       StringBuilder buff = new StringBuilder();
       String sep = "";
       for(String str : args) {
            buff.append(sep);
            buff.append(str);
            sep = ",";
       }
       String ret = String.format("%s(%s)", this.name, buff.toString());
       return (sign.equals("-")) ? String.format("not(%s)", ret) : ret;       
    }
    @Override
    public boolean equals(Object o) { 
  
        // If the object is compared with itself then return true   
        if (o == this) { 
            return true; 
        } 
  
        /* Check if o is an instance of Complex or not 
          "null instanceof [type]" also returns false */
        if (!(o instanceof Predicate)) { 
            return false; 
        } 
          
        // typecast o to Complex so that we can compare data members  
        Predicate c = (Predicate) o; 
          
        // Compare the data members and return accordingly  
        return c.toString().equals(this.toString()); 
    } 
}
