package edu.lsu.cct.cacuda;

import java.util.LinkedHashMap;
import java.util.Map;

class VarData {
	final String name;
	public VarData(String name) {
		this.name = name;
	}
	Map<String,String> attrs = new LinkedHashMap<String,String>();
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(name);
		sb.append(" :");
		for(Map.Entry<String,String> e : attrs.entrySet()) {
			sb.append(' ');
			sb.append(e.getKey());
			sb.append('=');
			sb.append('"');
			sb.append(e.getValue());
			sb.append('"');
		}
		sb.append('\n');
		return sb.toString();
	}
}