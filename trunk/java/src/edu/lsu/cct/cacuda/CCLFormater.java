package edu.lsu.cct.cacuda;

import java.io.File;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import edu.lsu.cct.piraha.Grammar;
import edu.lsu.cct.piraha.Matcher;
import edu.lsu.cct.piraha.examples.Calc;

public final class CCLFormater
{
  private final KernelData kd;
  private final int[] tile;
  private final File outFile;
  private final int[] stencil;
  private final Grammar g;
  public OutputGen outputgen;
  Calc calc = new Calc();
  String vname;
  public Map<String, Integer> forLoopVars = new HashMap<String, Integer>();

  public CCLFormater(KernelData kd, int[] tile, File outFile, int[] stencil,Grammar g)
  {
    this.kd = kd;
    this.tile = tile;
    this.outFile = outFile;
    this.stencil = stencil;
    this.g = g;
  }

  public String file() { return outFile.getName(); }

  public String date() { return new Date().toString(); }

  public String name_upper() { return outFile.getName().replace('.','_').toUpperCase(); }

  public String name() { return kd.name; }

  public String tile_x() { return Integer.toString(tile[0]); }

  public String tile_y() { return Integer.toString(tile[1]); }

  public String tile_z() { return Integer.toString(tile[2]); }

  public String stencil_xn() { return Integer.toString(stencil[0]); }

  public String stencil_xp() { return Integer.toString(stencil[1]); }

  public String stencil_yn() { return Integer.toString(stencil[2]); }

  public String stencil_yp() { return Integer.toString(stencil[3]); }

  public String stencil_zn() { return Integer.toString(stencil[4]); }

  public String stencil_zp() { return Integer.toString(stencil[5]); }
  
  public String conn() { return "\\\n"; }
  
  public String con() { return "\\"; }

  void parse_var(String par,Map<String,String> parMap) {
    Matcher m = g.matcher("par",par);
    if(m.matches()) {
      parMap.put(m.group(0).substring(), m.group(1).substring());
    } else {
      throw new RuntimeException("Error in "+par+" "+m.near());
    }
  }

  public String var_loop(String p1,String p2,String p3,String body)
  {
    Map<String,String> parMap = new HashMap<String,String>();
    parse_var(p1,parMap);
    parse_var(p2,parMap);
    parse_var(p3,parMap);
    return var_loop(parMap,body);
  }

  public String var_loop(String p1,String p2,String body)
  {
    Map<String,String> parMap = new HashMap<String,String>();
    parse_var(p1,parMap);
    parse_var(p2,parMap);
    return var_loop(parMap,body);
  }

  public String var_loop(String p1,String body)
  {
    Map<String,String> parMap = new HashMap<String,String>();
    parse_var(p1,parMap);
    return var_loop(parMap,body);
  }

  public String var_loop(String body)
  {
    Map<String,String> parMap = new HashMap<String,String>();
    return var_loop(parMap,body);
  }

  String var_loop(Map<String,String> parMap,String body)
  {
    StringBuilder sb = new StringBuilder();
    String delimit = null;
    for (VarData vd : kd.vars.values())
    {
      if (parMap.containsKey("intent"))
      {
        if (!parMap.get("intent").equals(vd.attrs.get("intent")))
        {
          continue;
        }
      }
      if (parMap.containsKey("cached"))
      {
        if (!parMap.get("cached").equals(vd.attrs.get("cached")))
        {
          continue;
        }
      }
      vname = vd.name;
      if (delimit != null)
        sb.append(delimit);
      sb.append(outputgen.replaceAll(this, body));
      if (delimit == null && parMap.containsKey("delimit"))
        delimit = parMap.get("delimit");
    }
    return sb.toString();
  }

  public String for_loop(String var, String lo, String hi, String body)
  {
    int lov = (int) calc.eval(outputgen.replaceAll(this, lo));
    int hiv = (int) calc.eval(outputgen.replaceAll(this, hi));
    int del = lov < hiv ? 1 : -1;
    StringBuilder sb = new StringBuilder();
    for (int i = lov; i != hiv; i += del)
    {
      forLoopVars.put(var, i);
      sb.append(outputgen.replaceAll(this, body));
    }
    //forLoopVars.remove(var);
    return sb.toString();
  }

  Map<String,String> vars = new HashMap<String,String>();
  public String set(String name,String val)
  {
    vars.put(name, val);
    return "";
  }
  
  public String var(String name)
  {
    Integer val = forLoopVars.get(name);
    if(val != null)
      return val.toString();
    String valStr = vars.get(name);
    if(valStr != null) {
      return valStr;
    }
    System.err.println("Warning: No value available for var '"+name+"'");
    return "";
  }
  
  public String unset(String name)
  {
    vars.remove(name);
    return "";
  }

  /**
   * This is the temporary variable used by the var_loop
   * @return
   */
  public String vname() {
    if(vname == null) {
      System.err.println("Warning %vname used outside of var_loop");
      return "";
    }
    return vname;
  }
}