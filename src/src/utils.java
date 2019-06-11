import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class utils {
    public static String[] alphabet = new String[]{"A","B","C","D","E","F","G",
                                                   "H","I","J","K","L","M","N",
                                                   "O","P","Q","R","S","T","U",
                                                   "V","W","X","Y","Z"};
    public static String getNextUnussedName(ArrayList<String> x_arrCurrentUsed)
    {
        for(int i = 0 ; i < alphabet.length ; i++)
            if(!x_arrCurrentUsed.contains(alphabet[i]))
                return alphabet[i];
        System.out.println("out of new variable name!");
        System.exit(-1);
        return null;
    }
    public static <T> Set<Set<T>> choose_k_of(Set<T> originalSet,int k)
    {
        Set<Set<T>> sets = utils.powerSet(originalSet);
        Set<Set<T>> sets_of_size_k = new HashSet<Set<T>>();
        for(Set<T> s:sets)
            if(s.size() == k)
                sets_of_size_k.add(s);
        return sets_of_size_k;
    }
    public static <T> boolean intersection(ArrayList<T> x_arr1, ArrayList<T> x_arr2)
    {
        boolean bRet = false;
        for(T t: x_arr1)
            if(x_arr2.contains(t))
                bRet = true;
        return bRet;
    }
    public static <T> Set<Set<T>> powerSet(Set<T> originalSet) 
    {
        Set<Set<T>> sets = new HashSet<Set<T>>();
        if (originalSet.isEmpty()) {
            sets.add(new HashSet<T>());
            return sets;
        }
        List<T> list = new ArrayList<T>(originalSet);
        T head = list.get(0);
        Set<T> rest = new HashSet<T>(list.subList(1, list.size())); 
        for (Set<T> set : powerSet(rest)) {
            Set<T> newSet = new HashSet<T>();
            newSet.add(head);
            newSet.addAll(set);
            sets.add(newSet);
            sets.add(set);
        }       
        return sets;
    }
    public static String toPrologList(ArrayList<Predicate> x_preds)
    {
        String strList ="[";
        for(int i = 0 ; i < x_preds.size() ; i++)
        {
            strList +=x_preds.get(i).toString();
            if(i < x_preds.size() - 1)
                strList +=",";
        }
        strList += "]";
        return strList;
    }
    public static int factorial(int n)
    {
        if(n == 0)
            return 1;
        return n * factorial(n - 1);
    }
    public static double choose_n_k(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n/2) {
        // choose(n,k) == choose(n,n-k), 
        // so this could save a little effort
        k = n - k;
    }

    double denominator = 1.0, numerator = 1.0;
    for (int i = 1; i <= k; i++) {
        denominator *= i;
        numerator *= (n + 1 - i);
    }
    return numerator / denominator; 
    }
    public static HashMap<String, Double> sortByValue(HashMap<String, Double> hm) 
    { 
        // Create a list from elements of HashMap 
        List<Map.Entry<String, Double> > list = new LinkedList<Map.Entry<String, Double> >(hm.entrySet()); 
  
        // Sort the list 
        Collections.sort(list, new Comparator<Map.Entry<String, Double> >() { 
            public int compare(Map.Entry<String, Double> o1,  
                               Map.Entry<String, Double> o2) 
            { 
                return (o2.getValue()).compareTo(o1.getValue()); 
            } 
        }); 
          
        // put data from sorted list to hashmap  
        HashMap<String, Double> temp = new LinkedHashMap<String, Double>(); 
        for (Map.Entry<String, Double> aa : list) { 
            temp.put(aa.getKey(), aa.getValue()); 
        } 
        return temp; 
    } 
}
