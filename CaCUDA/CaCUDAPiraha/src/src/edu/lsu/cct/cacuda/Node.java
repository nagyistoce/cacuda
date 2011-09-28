package edu.lsu.cct.cacuda;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import edu.lsu.cct.piraha.Group;

/**
 * Common routines for templates, groups and scripts to traverse 
 * through the tree, check the consistency, etc.
 * @author marqs
 *
 */
public class Node {
	
	public enum Scope {group, file, other};
	Scope scope = Scope.file;
	public String name;
	public String abspath = "";
	public Map<String, Node> triggers = new LinkedHashMap<String, Node>();
	public Map<String, Node> contains = new LinkedHashMap<String, Node>();
	public Map<String, String> attrs  = new LinkedHashMap<String, String>();
	public List<String> schedule      = new ArrayList<String>();
	public File file = null;
	public boolean evaluated = false;
	
	public Node(String name) {
		this.name = name;
	}
	
	public Node(Group m) throws Exception {
		this.parseInput(m);
	}

	public void put(Node node) throws Exception{
		if (contains.containsKey(node.name)){
			throw new Exception("The group already contains the key named: " + node.name);
		}
		contains.put(node.name, node);
	}
	
	public void addTrigger(Node node){
		triggers.put(node.name, node);
	}
	
	public boolean isLeaf(){
		return contains.isEmpty();
	}
	
	public boolean exists(){
		return file != null && file.exists();
	}
	
	public boolean checkConsistency(){
		if (! exists()) return false;
		for (Node node : contains.values()) {
			if(!node.checkConsistency())
				return false;
		}
		System.err.println("Node " + name + " checked consistency correct.");
		return true;
	}
	
	public static String join(String[] names, String del){ return join( names, del, 0); }
	public static String join(String[] names, String del, int i){
	    if (names.length <= i) return "";
	    StringBuffer buffer = new StringBuffer(names[i++]);
	    for(;i < names.length; i++) buffer.append(del).append(names[i]);
	    return buffer.toString();	
	}
	
	public Node find(String path){
		int start = "root".equals(name) ? 0 : 1;
		String nodes[] = path.split("/");
		if (nodes.length > 0 && (name.equals(nodes[0]) || start == 0)){
			if(nodes.length > start){
				String newpath = join(nodes, "/", start);
				Node ret = contains.get(nodes[start]);
				if(ret != null && (ret = ret.find(newpath)) != null)
					return ret;
				for (Node node : contains.values()){
					ret = node.find(newpath);
					if(ret != null)
						return ret;
				}
			}else return this;
		}
		return null;
	}
	
	
	public void findAll(String path, String reversepath, LinkedHashMap<String, Node> pathToNodes){
		int start = "root".equals(name) ? 0 : 1;
		String nodes[] = path.split("/");
		String newreversepath = "";
		if (!"root".equals(name)) newreversepath = reversepath + ("".equals(reversepath) ? "" : "/") + name;
		if (nodes.length > 0 && (start == 0 || "*".equals(nodes[0]) || name.equals(nodes[0]))){
			if(nodes.length > start){
				String newpath = join(nodes, "/", start); 
				for (Node node : contains.values())
					node.findAll(newpath, newreversepath, pathToNodes);
			}else pathToNodes.put(newreversepath, this);
		}
	}
	
	public void parseInput(Group m) throws Exception{
		String p = m.getPatternName();
		if ("name".equals(p)){
			this.name = m.substring();
		} 
		else throw new RuntimeException("Cannot parse pattern " + p + " near "+ m.near());
	}

	public void parseAttr(Group m, int i){ parseAttr(m, i, attrs); }
	public void parseAttr(Group m, int i, Map<String, String> to){
		if ("key".equals(m.group(i).getPatternName())){
			String val = m.group(i + 1).substring();
			if(('\'' == val.charAt(0) && '\'' == val.charAt(val.length() - 1)) ||
			   ('"'  == val.charAt(0) && '"'  == val.charAt(val.length() - 1)))
			   val = val.substring(1, val.length() - 1);
			to.put(m.group(i).substring(), val);
			if (to == attrs){
				if("scope".equals(m.group(i).substring())){
					if ("file".equals(m.group(i + 1).substring())){
						this.scope = Scope.file;
					} else if ("group".equals(m.group(i + 1).substring())){
						this.scope = Scope.group;
					} else {
						this.scope = Scope.other;
					}
				}
			}
		}else throw new RuntimeException("Pattern name in wrong place: "+ m.group(i).getPatternName());
	}
	
	public void parseTriggers(Group m){
		if(!"TRIGGERS".equals(m.getPatternName())) 
			throw new RuntimeException("Wrong group name " + m.getPatternName());
		for (int i = 0; i < m.groupCount(); i++)
		{
		  String p = m.group(i).getPatternName();
		  if ("key".equals(p)){
			  System.err.println("Attributes for triggering not implemented yet. Skipping: " + 
					  m.group(i).substring() + "=" + m.group(i + 1).substring());
			  i++;
		  } else if ("value".equals(p)){
			  triggers.put(m.group(i).substring(), null);
		  } else throw new RuntimeException("Unrecognized pattern name: " + m.getPatternName());
		}
	}
	
	public void parseSchedule(Group m){
		if(!"SCHEDULE".equals(m.getPatternName())) 
			throw new RuntimeException("Wrong group name " + m.getPatternName());
		for (int i = 0; i < m.groupCount(); i++)
		{
		  String p = m.group(i).getPatternName();
		  if ("key".equals(p)){
			  System.err.println("Attributes for triggering not implemented yet. Skipping: " + 
					  m.group(i).substring() + "=" + m.group(i + 1).substring());
			  i++;
		  } else if ("value".equals(p)){
			  schedule.add(m.group(i).substring());
		  } else throw new RuntimeException("Unrecognized pattern name: " + m.getPatternName());
		}
	}
	
	public void fillUp(TemplateGroup group, String abspath){
//		System.err.print("Template " + name + " triggers: ");
		for(Map.Entry<String, Node> entry : triggers.entrySet()){
			Node n = group.find(group.name + "/" + entry.getKey());
			if (n == null)
				throw new RuntimeException("Cannot find the template named: " + entry.getKey());
			entry.setValue(n);
//			System.err.print(n.name + ", ");
		}
//		System.err.println();
		
		if(file == null){
			String filename = attrs.get("file");
			if(filename == null) filename = name;
			file = new File(group.file, filename);
		}
		exists();
		this.abspath = (abspath.isEmpty() ? "" : abspath + "/") + name;
	}
}



