/**
 * This is class represents a template, i.e. a file to be evaluated.
 * @author marqs
 */

package edu.lsu.cct.cacuda;

import java.util.Map;

import edu.lsu.cct.piraha.Group;


public class Template extends Node {

	public boolean evaluated;
	
	public Template(String name){
		super(name);
	}
	
	public Template(Group m) throws Exception{
		super(m);
	}
	
	public boolean exists(){
		boolean ret = super.exists() && file.isFile();
		if (! ret) System.err.println("The template file does not exists: " + name);
		return ret;
	}
	
	public boolean checkConsistency(){
		if (!(super.checkConsistency() && contains.isEmpty()))
			throw new RuntimeException("Consistency check failed for template: " + name);
		return true;
	}
	
	
	public void parseInput(Group m) throws Exception{
		for (int i = 0; i < m.groupCount(); i++)
		{
		  String p = m.group(i).getPatternName();
		  if ("key".equals(p)){
			  parseAttr(m, i++);
		  } else if ("TRIGGERS".equals(p)){
			  parseTriggers(m.group(i));
		  }
		  else if ("SCHEDULE".equals(p)){
			  parseSchedule(m.group(i));
		  }else if (i == m.groupCount() - 1 && "dquote".equals(m.group(i).getPatternName()));
		   else super.parseInput(m.group(i));		  
		}
	}
	
	public Pair<String, String> produceFilenames(KernelData kd){ return produceFilenames(kd, false); }
	public Pair<String, String> produceFilenames(KernelData kd, boolean forGroup){
		String src     = file.getAbsolutePath();
		String names[] = src.split("[/.]");
		String dst;
		if(forGroup) dst = "CaKernel__" + abspath.replaceAll("\\$[^/]*$", "") + "." + names[names.length - 1];
		else dst = "CaKernel__" + abspath.replaceAll("\\$[^/]*$", "") + "__" + kd.name + "." + names[names.length - 1];
		
		dst = dst.replaceAll("/", "__");		
		return new Pair<String, String>(src, dst);
	}
	
	

	
}
