import java.util.ArrayList;
import java.util.List;

/**
 * This class represents an UtilityList as described in the paper.
 *
 * Copyright (c) 2008-2013 Philippe Fournier-Viger
 * 
 * This file is part of the SPMF DATA MINING SOFTWARE
 * (http://www.philippe-fournier-viger.com/spmf).
 *
 * SPMF is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPMF is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPMF.  If not, see <http://www.gnu.org/licenses/>.
 */
public class UtilityList {
	int item;  // the item
	int sumIutils = 0;  // the sum of item utilities
	int sumRutils = 0;  // the sum of remaining utilities
	List<Element> elements = new ArrayList<Element>();  // the elements
	
	public UtilityList(int item){
		this.item = item;
	}
	
	/**
	 * Method to add an element to this utility list and update the sums at the same time.
	 */
	public void addElement(Element element){
		sumIutils += element.iutils;
		sumRutils += element.rutils;
		elements.add(element);
	}
}
