import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import javafx.util.Pair;

public class LanguageBias {
    
    private HashMap<String,ArrayList<Pair<String,String>>> hModeH;
    private HashMap<String,ArrayList<Pair<String,String>>> hModeB;
    private HashMap<String,Integer> hPred_count;
    private HashMap<String,ArrayList<String>> hConstantVals;
    
    public LanguageBias(String x_strURL)
    {
        hModeH = new HashMap<>();
        hModeB = new HashMap<>();
        hPred_count = new HashMap<>();
        hConstantVals = new HashMap<>();
        try
        {
            File file = new File(x_strURL);
            FileReader fileReader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
	    String line;
	    while ((line = bufferedReader.readLine()) != null) 
            {
                if(line.startsWith("#modeh"))
                {
                    line = line.substring(line.indexOf("(")+1,line.lastIndexOf(")"));
                    String pred_name = line.substring(0,line.indexOf("("));
                    String arg_list =  line.substring(line.indexOf("(")+1, line.lastIndexOf(")"));
                    String[] arrArgs = arg_list.split(",");
                    for(String s:arrArgs)
                    {
                        Pair p = null;
                        String strType = s.substring(s.indexOf("(")+1,s.indexOf(")"));
                        if(s.startsWith("var("))
                            p = new Pair("var", strType);
                        else if(s.startsWith("const("))
                            p = new Pair("const", strType);                           
                        else
                        {
                            System.out.println(String.format("Error in line (%s) format",line));
                            System.exit(-1);
                        }                       
                        if(!hModeH.containsKey(pred_name))
                            hModeH.put(pred_name, new ArrayList<Pair<String,String>>());
                        hModeH.get(pred_name).add(p);
                    }
                    hPred_count.put(pred_name, 1);
                }
                else if(line.startsWith("#modeb"))
                {
                    line = line.substring(line.indexOf("(")+1,line.lastIndexOf(")"));
                    int max_pred_count = Integer.parseInt(line.substring(0,line.indexOf(",")));                    
                    line = line.substring(line.indexOf(",")+1);
                    String pred_name = line.substring(0,line.indexOf("("));                    
                    hPred_count.put(pred_name, max_pred_count);
                    line = line.substring(line.indexOf("(")+1,line.lastIndexOf(")"));
                    String[] arrArgs = line.split(",");
                    for(String s:arrArgs)
                    {
                        Pair p = null;
                        String strType = s.substring(s.indexOf("(")+1,s.indexOf(")"));
                        if(s.startsWith("var("))
                            p = new Pair("var", strType);
                        else if(s.startsWith("const("))
                            p = new Pair("const", strType);                           
                        else
                        {
                            System.out.println(String.format("Error in line (%s) format",line));
                            System.exit(-1);
                        }
                       
                        if(!hModeB.containsKey(pred_name))
                            hModeB.put(pred_name, new ArrayList<Pair<String,String>>());
                        hModeB.get(pred_name).add(p);
                    }
                } 
               else if(line.startsWith("#constant"))
               {
                   line = line.substring(line.indexOf("(")+1,line.lastIndexOf(")"));
                   String[] const_pair = line.split(",");
                   if(!hConstantVals.containsKey(const_pair[0]))
                       hConstantVals.put(const_pair[0], new ArrayList<String>());
                   hConstantVals.get(const_pair[0]).add(const_pair[1]);
               }
            }
            fileReader.close();
	}
        catch (IOException e) 
        {
            e.printStackTrace();
            System.exit(-1);
        }
    }
    public double getDifferentLiteralsBits()
    {
        double dTotalCount = 0;
        for(String s:hModeB.keySet())
        {
            int temp = 1;
            ArrayList<Pair<String,String>> arrArguments = hModeB.get(s);
            for(Pair<String,String> p:arrArguments)
            {
                if(p.getKey().equals("const"))
                    temp *= hConstantVals.get(p.getValue()).size();
            }
            dTotalCount += temp;
        }
        return Math.ceil(Math.log(dTotalCount) / Math.log(2));
    }
    private ArrayList<Pair<String,String>> getVars_and_types(Clause x_cl)
    {
        if(!hModeH.containsKey(x_cl.Head.name))
        {
            System.out.printf("invalid clause head %s\r\n",x_cl.Head.name);
            System.exit(-1);
        }
        ArrayList<Pair<String,String>> arrVarTypes = new ArrayList<>();
        ArrayList<Pair<Integer,String>> arrIndexVar = x_cl.Head.getVars_withIndex();
        ArrayList<Pair<String,String>> arrVarType = hModeH.get(x_cl.Head.name);
        for(Pair p:arrIndexVar)
        {
            int arg_index = (int)p.getKey();
            String strVar_Id = (String)p.getValue();
            String strVar_type = arrVarType.get(arg_index).getValue();
            arrVarTypes.add(new Pair(strVar_Id,strVar_type));            
        }
        for(Predicate b:x_cl.Body)
        {
            if(hModeH.containsKey(b.name)) // this is a recursive rule
                arrVarType = hModeH.get(b.name);
            else if(hModeB.containsKey(b.name))
                arrVarType = hModeB.get(b.name);
            else
            {
                System.out.printf("invalid predicate name %s\r\n", b.name);
                System.exit(-1);
            }
            arrIndexVar = b.getVars_withIndex();
            for(Pair p:arrIndexVar)
            {
                int arg_index = (int)p.getKey();
                String strVar_Id = (String)p.getValue();
                String strVar_type = arrVarType.get(arg_index).getValue();
                arrVarTypes.add(new Pair(strVar_Id,strVar_type));            
            }  
        }
        return arrVarTypes;        
    }
    public ArrayList<Clause> refine(Clause x_cl)
    {
        ArrayList<Clause> arrRefinded = new ArrayList<>();
        for(String pred:hModeB.keySet())
        {
            ArrayList<Clause> arrTemp = refine_using_predicate(x_cl, pred);
            arrRefinded.addAll(arrTemp);           
        }
        return arrRefinded;
    }
    private ArrayList<Clause> refine_using_predicate(Clause x_cl,String x_strPredName)
    {
        ArrayList<Clause> arrRefinedClauses = new ArrayList<>();
        if(hPred_count.get(x_strPredName) == x_cl.getPredicateCount(x_strPredName))
            return arrRefinedClauses;
                
        ArrayList<Pair<String,String>> arrPredicateTemplate = null;
        if(hModeB.containsKey(x_strPredName))
            arrPredicateTemplate = hModeB.get(x_strPredName);
        else if(hModeB.containsKey(x_strPredName))
            arrPredicateTemplate = hModeH.get(x_strPredName);
        
        ArrayList<Pair<String,String>> arrVar_and_types = this.getVars_and_types(x_cl);
        ArrayList<ArrayList<String>> arrArgumentVariableOptions = new ArrayList<ArrayList<String>>();
        for(int i = 0 ; i < arrPredicateTemplate.size() ; i++ )
        {
            ArrayList<String> arrOption = new ArrayList<>();
            for(Pair<String,String> p:arrVar_and_types)
            {
                if(arrPredicateTemplate.get(i).getKey().equals("var") &&
                        p.getValue().equals(arrPredicateTemplate.get(i).getValue()) && !arrOption.contains(p.getKey()))
                {
                    arrOption.add(p.getKey());
                }
            }
            if(arrPredicateTemplate.get(i).getKey().equals("var"))
                arrOption.add("_");
            if(arrPredicateTemplate.get(i).getKey().equals("const"))
            {
                if(!hConstantVals.containsKey(arrPredicateTemplate.get(i).getValue()))
                {
                    System.out.printf("Invalid constant type %s in Predicate %s\r\n",arrPredicateTemplate.get(i).getValue(),x_strPredName);
                    System.exit(-1);
                }
                ArrayList<String> arrAllVals = hConstantVals.get(arrPredicateTemplate.get(i).getValue());
                for(String s:arrAllVals)
                    arrOption.add(s);
            }
            
            arrArgumentVariableOptions.add(arrOption);
        }
        ///////////////
        ArrayList<ArrayList<String>> arrTemp = getAllCombinations(arrArgumentVariableOptions, new ArrayList<ArrayList<String>>(), 0);
        for(ArrayList<String> arr: arrTemp)
        {
            ArrayList<String> arrCurrentVars = x_cl.getVars();
            String strArgList = "";
            for(int i = 0 ; i < arr.size() ; i++)
            {
                if(arr.get(i).equals("_")) // define a new variable
                {
                    String v = utils.getNextUnussedName(arrCurrentVars);
                    arrCurrentVars.add(v);
                    strArgList += v;
                }
                else
                {
                    strArgList += arr.get(i);
                }
                if(i < arr.size() - 1)
                    strArgList += ",";             
            }
            String strNewPred = String.format("%s(%s)",x_strPredName,strArgList);
            Predicate p = new Predicate(strNewPred);
            /////////////////
            if(!utils.intersection(p.getVarNames(),x_cl.getVars()))
               continue;
            
            ////////////////
            Clause cl = new Clause(x_cl.toString());
            cl.addPredicate(p);
            arrRefinedClauses.add(cl);
        }
        return arrRefinedClauses;
    }
    private ArrayList<ArrayList<String>> getAllCombinations(ArrayList<ArrayList<String>> x_arrInput,ArrayList<ArrayList<String>> x_arrOutput,int x_nCurrentLayer)
    {
        if(x_nCurrentLayer == 0)
        {
            for(String s: x_arrInput.get(0))
            {
                ArrayList<String> arrOutItem = new ArrayList<>();
                arrOutItem.add(s);
                x_arrOutput.add(arrOutItem);
            }
            return getAllCombinations(x_arrInput, x_arrOutput, 1);
        }
        if(x_nCurrentLayer == x_arrInput.size())
            return x_arrOutput;
        ArrayList<ArrayList<String>> arrOutput = new ArrayList<>();
        for(String s:x_arrInput.get(x_nCurrentLayer))
        {
            for(ArrayList<String> arr:x_arrOutput)
            {
                ArrayList<String> arrTemp = (ArrayList<String>)arr.clone();
                arrTemp.add(s);
                arrOutput.add(arrTemp);
            }
        }
        if(x_nCurrentLayer == x_arrInput.size() - 1)
            return arrOutput;
        else
            return getAllCombinations(x_arrInput,arrOutput,++x_nCurrentLayer);        
    }
    public Clause getGoalClause()
    {
        String pred_name = hModeH.keySet().toArray()[0].toString();
        int arity = hModeH.get(pred_name).size();
        String arg_list = "";
        for(int i = 0 ; i < arity ; i++)
        {
            arg_list += utils.alphabet[i];
            if( i < arity - 1)
                arg_list += ",";
        }
        return new Clause(String.format("%s(%s)",pred_name,arg_list));
    }
}
