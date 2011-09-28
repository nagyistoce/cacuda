/**
 * Groups of templates. Either for architecture or for optimization purposes.
 * Groups can contain other groups, templates or scripts.
 * @author marqs
 */
package edu.lsu.cct.cacuda;

import edu.lsu.cct.piraha.Group;

/**
 * The class contains group of templates. In the future I think this should be a place to prepare some optimizations, etc.
 * @author marqs
 * 
 */

public class TemplateGroup extends Node {

	public TemplateGroup(String name){
		super(name);
	}
	
	public TemplateGroup(Group m) throws Exception{
		super(m);
	}

	public boolean exists(){
		return super.exists() && file.isDirectory();
	}
	
	public boolean checkConsistency(){
		if(!super.checkConsistency())
			throw new RuntimeException("Consistency check failed for group: " + this.name);
		if(contains.isEmpty())
			throw new RuntimeException("Consistency check failed for group: " + this.name + " because the group is empty");
		return true;
	}
	
	public void parseInput(Group m) throws Exception{
	    for (int i = 0; i < m.groupCount(); i++)
	    {
	    	String p = m.group(i).getPatternName();
	    	if ("TEMPLATE_GROUP".equals(p)) {
	    		put(new TemplateGroup(m.group(i)));
			} else if ("TEMPLATE".equals(p)) {
				put(new Template(m.group(i)));
			} else if ("SCRIPT".equals(p)) {
				put(new Script(m.group(i)));
			} else if ("key".equals(p)){
				parseAttr(m, i++);
			} else super.parseInput(m.group(i));
	    }
	    if ("root".equals(name)){
	    	abspath = "";
	    	fillUp(this, abspath);
	    	checkConsistency();	
	    }
	}
	
	public void fillUp(TemplateGroup group, String abspath){
		if (!"root".equals(name))
			super.fillUp(group, abspath);
			
		for(Node n : contains.values())
			n.fillUp(this, this.abspath);
	}
}
