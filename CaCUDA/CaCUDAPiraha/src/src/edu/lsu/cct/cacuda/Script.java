/**
 * Scripts. We will need that for OpenCL...
 * @author marqs 
 */
package edu.lsu.cct.cacuda;

import edu.lsu.cct.piraha.Group;

public class Script extends Node {

	public static final int EXECUTE_PREPIRAHA = 1;
	public static final int EXECUTE_POSTPIRAHA = 2;
	
	/*** TODO here we will have to edit the schedule that it would run 
	 *        before each compilation (not configuration!!!). Jian some help?
	 */
	public static final int EXECUTE_PRECOMPILATION = 4;
	/// TODO some more possibilities? 
	
	public void parseInput(Group m) throws IllegalAccessException{
		throw new IllegalAccessException("Not implemented yet");
	}
	
	public Script(String name){
		super(name);
	}
	
	public Script(Group m) throws Exception{
		super(m) ;
	}
	
	public boolean exists(){
		return super.exists() && file.isFile() && file.canExecute();
	}
	
	public boolean checkConsistency(){
		return super.checkConsistency() && contains.isEmpty();
	}
	
}
