import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * A simple implementation of the TKO algorithm without some of the 
 * optimizations described in the paper.
 * 
 * @author Philippe Fournier-Viger et al.
 */
public class AlgoTKO_Basic {

	/** the time the algorithm terminated */
	long totalTime = 0; 
	
	/** the number of HUI generated  */
	int huiCount = 0; 

	/** the k parameter */
	int k = 0;
	
	/** the internal min utility variable */
	long minutility = 0; 

	/** the top k rules found until now */
	PriorityQueue<Itemset> kItemsets; 

	/** We create a map to store the TWU of each item */
	final Map<Integer, Integer> mapItemToTWU = new HashMap<Integer, Integer>();

	/** this class represent an item and its utility in a transaction */
	class Pair {
		/** an item */
		int item = 0;
		
		/** the utility of the item */
		int utility = 0;
	}

	/** 
	 * Constructor
	 */
	public AlgoTKO_Basic() {

	}

	/**
	 * Run the algorithm
	 * @param input the input file path
	 * @param output the output file path
	 * @param k the parameter k
	 * @throws IOException if an error occur for reading/writing to file.
	 */
        public void runAlgorithm(ArrayList<String> arrInput, int k) throws IOException {
            MemoryLogger.getInstance().reset();
            long startTimestamp = System.currentTimeMillis();
            this.minutility = 1;
            this.k = k;

            this.kItemsets = new PriorityQueue<Itemset>();
            String thisLine;

            // for each line (transaction)
            for(int j = 0 ; j < arrInput.size() ; j++)
            {
                thisLine = arrInput.get(j);
                // if the line is  a comment, is  empty or is a
                // kind of metadata
                if (thisLine.isEmpty() == true ||	thisLine.charAt(0) == '#' 
                                || thisLine.charAt(0) == '%'
                        || thisLine.charAt(0) == '@') {
                        continue;
                }

                String split[] = thisLine.split(":");
                // the list of items
                String items[] = split[0].split(" ");
                // the transaction utility
                int transactionUtility = Integer.parseInt(split[1]);
                // for each item, we add the transaction utility to its TWU
                for (int i = 0; i < items.length; i++) {
                        Integer item = Integer.parseInt(items[i]);
                        // get the current TWU
                        Integer twu = mapItemToTWU.get(item);
                        // update the twu
                        twu = (twu == null) ? transactionUtility : twu
                                        + transactionUtility;
                        mapItemToTWU.put(item, twu);
                }
            }

            // CREATE A LIST TO STORE THE ITEMS WITH TWU >= MIN_UTILITY.
            List<UtilityList> listItems = new ArrayList<UtilityList>();

            // CREATE A MAP TO STORE THE UTILITY LIST FOR EACH ITEM.
            Map<Integer, UtilityList> mapItemToUtilityList = new HashMap<Integer, UtilityList>(
                            100000);
            // For each item
            for (Integer item : mapItemToTWU.keySet()) {
                    UtilityList uList = new UtilityList(item);
                    // add the item to the list of high TWU items
                    listItems.add(uList);
                    // create an empty Utility List that we will fill later.
                    mapItemToUtilityList.put(item, uList);
            }
            // SORT THE LIST OF HIGH TWU ITEMS IN ASCENDING ORDER
            Collections.sort(listItems, new Comparator<UtilityList>() {
                    public int compare(UtilityList o1, UtilityList o2) {
                            int compare = mapItemToTWU.get(o1.item)
                                            - mapItemToTWU.get(o2.item);
                            if (compare == 0) {
                                    // return (ascendingOrder) ? o1.item - o2.item: o2.item -
                                    // o1.item;
                                    return o1.item - o2.item;
                            }
                            return compare;
                    }
            });

            // SECOND DATABASE PASS TO CONSTRUCT THE UTILITY LISTS
            // OF 1-ITEMSETS HAVING TWU >= minutil (promising items)

            int tid = 0;
            // for each line (transaction)
            for(int j = 0 ; j < arrInput.size() ; j++)
            {
                thisLine = arrInput.get(j);
                String split[] = thisLine.split(":");
                // the list of items
                String items[] = split[0].split(" ");
                // the list of utility values
                String utilityValues[] = split[2].split(" ");

                int remainingUtility = 0;

                // Create a list to store items
                List<Pair> revisedTransaction = new ArrayList<Pair>();
                // for each item
                for (int i = 0; i < items.length; i++) {
                        // / convert values to integers
                        Pair pair = new Pair();
                        pair.item = Integer.parseInt(items[i]);
                        pair.utility = Integer.parseInt(utilityValues[i]);
                        revisedTransaction.add(pair);
                        remainingUtility += pair.utility;
                }

                Collections.sort(revisedTransaction, new Comparator<Pair>() {
                        public int compare(Pair o1, Pair o2) {
                                return compareItems(o1.item, o2.item);
                        }
                });

                // for each item left in the transaction
                for (Pair pair : revisedTransaction) {

                        // subtract the utility of this item from the remaining
                        // utility
                        remainingUtility = remainingUtility - pair.utility;

                        // get the utility list of this item
                        UtilityList utilityListOfItem = mapItemToUtilityList
                                        .get(pair.item);

                        // Add a new Element to the utility list of this item
                        // corresponding to this transaction
                        Element element = new Element(tid, pair.utility,
                                        remainingUtility);

                        utilityListOfItem.addElement(element);
                }
                tid++; // increase tid number for next transaction
            }


//		System.out.println(minutility);

            // check the memory usage
            MemoryLogger.getInstance().checkMemory();

            // Mine the database recursively
            search(new int[0], null, listItems);

            // check the memory usage again and close the file.
            MemoryLogger.getInstance().checkMemory();
            totalTime = (System.currentTimeMillis() - startTimestamp) / 1000;
            
        }
	/**
	 * This is the recursive method to find all high utility itemsets. It writes
	 * the itemsets to the output file.
	 * 
	 * @param prefix
	 *            This is the current prefix. Initially, it is empty.
	 * @param pUL
	 *            This is the Utility List of the prefix. Initially, it is
	 *            empty.
	 * @param ULs
	 *            The utility lists corresponding to each extension of the
	 *            prefix.
	 * @param minUtility
	 *            The minUtility threshold.
	 * @throws IOException
	 */
	private void search(int[] prefix, UtilityList pUL, List<UtilityList> ULs)
			throws IOException {
		MemoryLogger.getInstance().checkMemory();

		// For each extension X of prefix P
		for (int i = 0; i < ULs.size(); i++) {
			UtilityList X = ULs.get(i);

			// If pX is a high utility itemset.
			// we save the itemset: pX
			if (X.sumIutils >= minutility) {
				writeOut(prefix, X.item, X.sumIutils);
			}

			// If the sum of the remaining utilities for pX
			// is higher than minUtility, we explore extensions of pX.
			// (this is the pruning condition)
			if (X.sumRutils + X.sumIutils >= minutility) {
				// This list will contain the utility lists of pX extensions.
				List<UtilityList> exULs = new ArrayList<UtilityList>();
				// For each extension of p appearing
				// after X according to the ascending order
				for (int j = i + 1; j < ULs.size(); j++) {
					UtilityList Y = ULs.get(j);
					// we construct the extension pXY
					// and add it to the list of extensions of pX
					exULs.add(construct(pUL, X, Y));
				}
				// We create new prefix pX
				int[] newPrefix = new int[prefix.length + 1];
				System.arraycopy(prefix, 0, newPrefix, 0, prefix.length);
				newPrefix[prefix.length] = X.item;

				// We make a recursive call to discover all itemsets with the
				// prefix pX
				search(newPrefix, X, exULs);
			}

		}
	}

	/**
	 * Method to write a high utility itemset to the output file.
	 * @param a prefix itemset
	 * @param an item to be appended to the prefix
	 * @param utility the utility of the prefix concatenated with the item
	 */
	private void writeOut(int[] prefix, int item, long utility) {
		Itemset itemset = new Itemset(prefix, item, utility);
		kItemsets.add(itemset);
		if (kItemsets.size() > k) {
			if (utility > this.minutility) {
				Itemset lower;
				do {
					lower = kItemsets.peek();
					if (lower == null) {
						break; // / IMPORTANT
					}
					kItemsets.remove(lower);
				} while (kItemsets.size() > k);
				this.minutility = kItemsets.peek().utility;
//				System.out.println(this.minutility);
			}
		}
	}

	/**
	 * This method constructs the utility list of pXY
	 * @param P :  the utility list of prefix P.
	 * @param px : the utility list of pX
	 * @param py : the utility list of pY
	 * @return the utility list of pXY
	 */
	private UtilityList construct(UtilityList P, UtilityList px, UtilityList py) {
		// create an empy utility list for pXY
		UtilityList pxyUL = new UtilityList(py.item);
		// for each element in the utility list of pX
		for(Element ex : px.elements){
			// do a binary search to find element ey in py with tid = ex.tid
			Element ey = findElementWithTID(py, ex.tid);
			if(ey == null){
				continue;
			}
			// if the prefix p is null
			if(P == null){
				// Create the new element
				Element eXY = new Element(ex.tid, ex.iutils + ey.iutils, ey.rutils);
				// add the new element to the utility list of pXY
				pxyUL.addElement(eXY);
				
			}else{
				// find the element in the utility list of p wih the same tid
				Element e = findElementWithTID(P, ex.tid);
				if(e != null){
					// Create new element
					Element eXY = new Element(ex.tid, ex.iutils + ey.iutils - e.iutils,
								ey.rutils);
					// add the new element to the utility list of pXY
					pxyUL.addElement(eXY);
				}
			}	
		}
		// return the utility list of pXY.
		return pxyUL;
	}
	
	/**
	 * Do a binary search to find the element with a given tid in a utility list
	 * @param ulist the utility list
	 * @param tid  the tid
	 * @return  the element or null if none has the tid.
	 */
	private Element findElementWithTID(UtilityList ulist, int tid){
		List<Element> list = ulist.elements;
		
		// perform a binary search to check if  the subset appears in  level k-1.
        int first = 0;
        int last = list.size() - 1;
       
        // the binary search
        while( first <= last )
        {
        	int middle = ( first + last ) >>> 1; // divide by 2

            if(list.get(middle).tid < tid){
            	first = middle + 1;  //  the itemset compared is larger than the subset according to the lexical order
            }
            else if(list.get(middle).tid > tid){
            	last = middle - 1; //  the itemset compared is smaller than the subset  is smaller according to the lexical order
            }
            else{
            	return list.get(middle);
            }
        }
		return null;
	}

	/**
	 * Write the result to a file
	 * @param path the output file path
	 * @throws IOException if an exception for reading/writing to file
	 */
        public ArrayList<String> getResulted_HUI()
        {
            ArrayList<String> arrOutput = new ArrayList<>();
            Iterator<Itemset> iter = kItemsets.iterator();
            while (iter.hasNext()) 
            {
                StringBuffer buffer = new StringBuffer();
                Itemset itemset = (Itemset) iter.next();

                // append the prefix
                for (int i = 0; i < itemset.getItemset().length; i++) 
                {
                    buffer.append(itemset.getItemset()[i]);
                    buffer.append(' ');
                }
                buffer.append(itemset.item);

                // append the utility value
                //buffer.append(" #UTIL: ");
                //buffer.append(itemset.utility);
                if(itemset.utility > 0)
                    arrOutput.add(buffer.toString());                
            }
            return arrOutput;
        }
	public void writeResultTofile(String path) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(path));
		Iterator<Itemset> iter = kItemsets.iterator();
		while (iter.hasNext()) {
			StringBuffer buffer = new StringBuffer();
			Itemset itemset = (Itemset) iter.next();
			
			// append the prefix
			for (int i = 0; i < itemset.getItemset().length; i++) {
				buffer.append(itemset.getItemset()[i]);
				buffer.append(' ');
			}
			buffer.append(itemset.item);
			
			// append the utility value
			buffer.append(" #UTIL: ");
			buffer.append(itemset.utility);
			
			// write to file
			writer.write(buffer.toString());
			if(iter.hasNext()){
				writer.newLine();
			}
		}
		writer.close();
	}


	private int compareItems(int item1, int item2) {
		int compare = mapItemToTWU.get(item1) - mapItemToTWU.get(item2);
		// if the same, use the lexical order otherwise use the TWU
		return (compare == 0) ? item1 - item2 : compare;
	}

	/**
	 * Print statistics about the latest execution to System.out.
	 */
	public void printStats() {
		System.out
				.println("=============  TKO-BASIC - v.2.28 =============");
		System.out
				.println(" High-utility itemsets count : " + kItemsets.size());
		System.out.println(" Total time ~ " + totalTime
				+ " s");
		System.out.println(" Memory ~ " + MemoryLogger.getInstance().getMaxMemory() + " MB");
		System.out.println("===================================================");
	}
}