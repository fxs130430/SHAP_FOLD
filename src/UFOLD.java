import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import org.jpl7.*;
import javafx.util.Pair;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class UFOLD {
    public static void show_format()
    {
        System.out.println("Input Error: Missing Parameters...");
        System.out.println("Valid Arguments:");
        System.out.println("java ufold -mode fold training_data_file.pl bias_file.txt");
        System.out.println("java ufold -mode shapfold <dataset prefix>");
    }
    public static void main(String[] args) {
        // TODO code application logic here
        //FOLD f = new FOLD("diabetes.pl","diabetes.txt");
        //FOLD f = new FOLD("cancer_train_2.pl","cancer.txt");
        //FOLD f = new FOLD("flies.pl","flies.txt");
        
        //FOLD f = new FOLD("kidney_train.pl","kidney_bias.txt");
        //FOLD f = new FOLD("breastw_train.pl","breastw_bias.txt");
        //FOLD f = new FOLD("thyroid_train.pl","thyroid_bias.txt");
        //FOLD f = new FOLD("acute_train.pl","acute_bias.txt");
        //FOLD f = new FOLD("custom_2.pl","custom_bias.txt");
        //FOLD f = new FOLD("autism_train.pl","autism_bias.txt");
        //FOLD f = new FOLD("surgery_train.pl","surgery_bias.txt");
        //FOLD f = new FOLD("sonar_train.pl","sonar_bias.txt");
        //FOLD f = new FOLD("ionosphere_train.pl","ionosphere_bias.txt");
        //FOLD f = new FOLD("ecoli_train.pl","ecoli_bias.txt");
        //FOLD f = new FOLD("wine_train.pl","wine_bias.txt");       
        //FOLD f = new FOLD("voting_train.pl","voting_bias.txt");       
        //FOLD f = new FOLD("credit_train.pl","credit_bias.txt");       
        //FOLD f = new FOLD("loan_train.pl","loan_bias.txt");
        //FOLD f = new FOLD("fold_income_train.pl","fold_income_bias.txt");
        
        //FOLD f = new FOLD("svm_heart_train.pl","svm_heart_bias.txt");
        //FOLD f = new FOLD("voting_train.pl","voting_bias.txt");
        
        //FOLD f = new FOLD("heart_train.pl","heart_bias.txt");
        //FOLD f = new FOLD("breastw_train.pl","breastw_bias.txt","breastw_shap_values.txt");
        //FOLD f = new FOLD("custom_train.pl","custom_bias.txt");
        //FOLD f = new FOLD("mushroom_train.pl","mushroom_bias.txt");
        //FOLD f = new FOLD("ionosphere_train.pl","ionosphere_bias.txt");
        //FOLD f = new FOLD("krkp_train.pl","krkp_bias.txt");
        //FOLD f = new FOLD("sonar_train.pl","sonar_bias.txt");
        
        
        //HUI_FOLD f = new HUI_FOLD("heart_train.pl","heart_bias.txt","heart_itemset_mining.txt","heart_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("breastw_train.pl","breastw_bias.txt","breastw_itemset_mining.txt","breastw_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("autism_train.pl","autism_bias.txt","autism_itemset_mining.txt","autism_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("custom_train.pl","custom_bias.txt","custom_itemset_mining.txt","custom_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("kidney_train.pl","kidney_bias.txt","kidney_itemset_mining.txt","kidney_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("credit_train.pl","credit_bias.txt","credit_itemset_mining.txt","credit_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("voting_train.pl","voting_bias.txt","voting_itemset_mining.txt","voting_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("lymph_train.pl","lymph_bias.txt","lymph_itemset_mining.txt","lymph_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("mushroom_train.pl","mushroom_bias.txt","mushroom_itemset_mining.txt","mushroom_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("ionosphere_train.pl","ionosphere_bias.txt","ionosphere_itemset_mining.txt","ionosphere_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("krkp_train.pl","krkp_bias.txt","krkp_itemset_mining.txt","krkp_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("acute_train.pl","acute_bias.txt","acute_itemset_mining.txt","acute_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("sonar_train.pl","sonar_bias.txt","sonar_itemset_mining.txt","sonar_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("monks3_train.pl","monks3_bias.txt","monks3_itemset_mining.txt","monks3_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("monks2_train.pl","monks2_bias.txt","monks2_itemset_mining.txt","monks2_index2colname.txt");
        //HUI_FOLD f = new HUI_FOLD("cars_train.pl","cars_bias.txt","cars_itemset_mining.txt","cars_index2colname.txt");
        
        //HUI_FOLD f = new HUI_FOLD("emotions_train.pl","emotions_bias.txt","emotions_itemset_mining.txt","emotions_index2colname.txt");
        //LIME_FOLD f = new LIME_FOLD("emotions_train.pl", "emotions_bias.txt", "emotions_shap_values.txt");
        //FOLD f = new FOLD("emotions_train.pl", "emotions_bias.txt");
        //HUI_FOLD f = new HUI_FOLD("emotions_train.pl","emotions_bias.txt","emotions_itemset_mining.txt","emotions_index2colname.txt");
        //try{
        //f.foil_run();
        //f.post_processing();
        //f.Display_Hypothesis();
        //}
        //catch(Exception ex)
        //{
        //    ex.printStackTrace();
        //}
        
        
        //System.exit(0);
        
        Object f = null;
        if(args.length <= 2)
        {
            UFOLD.show_format();
        }
        else if(args[0].equals("-mode"))
        {
            if(args[1].equals("fold"))
            {
                if(args.length != 4)
                {
                    UFOLD.show_format();
                }
                else
                {
                    String training_file = args[2];
                    String bias_file = args[3];
                    f = new FOLD(training_file,bias_file);
                    ((FOLD)f).foil_run();
                    ((FOLD)f).post_processing();
                    ((FOLD)f).Display_Hypothesis();
                }                    
            }
            else if(args[1].equals("shapfold"))
            {
                if(args.length != 3)
                {
                    UFOLD.show_format();
                }
                else
                {
                    String prefix = args[2];
                    String training_file = String.format("{0}_train.pl",prefix);
                    String bias_file = String.format("{0}_bias.txt",prefix);
                    String itemset_file = String.format("{0}_itemset_mining.txt",prefix);
                    String index2col_file = String.format("{0}_index2colname.txt",prefix);
                    f = new HUI_FOLD(training_file,bias_file,itemset_file,index2col_file);
                    try
                    {
                        ((HUI_FOLD)f).foil_run();
                        ((HUI_FOLD)f).post_processing();
                        ((HUI_FOLD)f).Display_Hypothesis();
                    }
                    catch(Exception ex)
                    {
                        ex.printStackTrace();
                    }
                }
            }
        }
    }
    
}
