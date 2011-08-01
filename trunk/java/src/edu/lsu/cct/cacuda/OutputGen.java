package edu.lsu.cct.cacuda;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.lsu.cct.piraha.Grammar;
import edu.lsu.cct.piraha.Matcher;

public class OutputGen
{
  public static final String DEFAULT_TEMPLATE = "$default";
  Grammar g = new Grammar();
  Map<String, String> map = new HashMap<String, String>();
  StringBuilder buf = new StringBuilder();
  String templateName = DEFAULT_TEMPLATE;

  public OutputGen()
  {
    g.compile("template", "%TEMPLATE\\({-w}\"{dquote}\"{-w}\\)");
    g.compile("input", "\"{dquote}\"|'{squote}'|{val}");
    g.compile("squote", "(\\\\[^]|[^\\\\'])*");
    g.compile("dquote", "(\\\\[^]|[^\\\\\"])*");
    g.compile("val","{name}|{num}");
    g.compile("name", "[a-zA-Z0-9_]+");
    g.compile("vname","[a-zA-Z][a-zA-Z0-9_]*");
    g.compile("w","[ \t\r\n]*");
    g.compile("num","[0-9]+");
    g.compile("block","%\\[{blockin}\\]%");
    g.compile("blockin","(%\\[{blockin}\\]%|[^\\]]|\\](?!%))*");
    g.compile("format", "%\\{{vname}\\}|%{vname}(\\(({-w}{input}({-w},{-w}{input})*|){-w}\\)|)({-w}{block}|)");
  }

  public void readSource(BufferedReader br) throws IOException
  {
    for (String s = br.readLine(); s != null; s = br.readLine())
    {
      addLine(s);
    }
    finishLines();
  }

  private void addLine(String s)
  {
    Matcher m = g.matcher("template", s);
    if (m.match(0))
    {
      if (templateName != null)
        map.put(templateName, buf.toString());
      templateName = m.group(0).substring();
      buf = new StringBuilder();
    } else
    {
      buf.append(s);
      buf.append('\n');
    }
  }

  private void finishLines()
  {
    map.put(templateName, buf.toString());
  }

  public void output(File f, String templateName, Object formatObject) throws IOException
  {
    FileWriter fw = new FileWriter(f);
    BufferedWriter bw = new BufferedWriter(fw);
    output(bw,templateName,formatObject);
    bw.flush();
  }
  
  public String outputLiteral(String templateStr, Object formatObject) {
    StringWriter sw = new StringWriter();
    try
    {
      outputLiteral(sw,templateStr,formatObject);
    } catch (IOException e)
    {
      System.err.println(e.getLocalizedMessage());
    }
    return sw.toString();
  }

  public String output(String templateName, Object formatObject) {
    StringWriter sw = new StringWriter();
    try
    {
      output(sw,templateName,formatObject);
    } catch (IOException e)
    {
      System.err.println(e.getLocalizedMessage());
    }
    return sw.toString();
  }
  
  public void output(Writer w, String templateName, Object formatObject) throws IOException
  {
    String s = map.get(templateName);
    if (s == null)
      throw new RuntimeException("No such template " + templateName);
    outputLiteral(w, s, formatObject);
  }
  
  public void outputLiteral(Writer w, String templateStr, Object formatObject) throws IOException
  {
    Class<?> formatClass = formatObject.getClass();
    try
    {
      Field f = formatClass.getField("outputgen");
      if(f != null)
        f.set(formatObject, this);
    } catch (Exception e)
    {
    }
    Map<String,Method> methods = new HashMap<String,Method>();
    for(Method m : formatClass.getMethods()) {
      boolean valid = true;
      if(m.getReturnType() != String.class)
        continue;
      for(Class<?> p : m.getParameterTypes()) {
        if(p != String.class) {
          valid = false;
          break;
        }
      }
      if(valid)
        methods.put(m.getName()+"%"+m.getParameterTypes().length,m);
    }
    //System.out.println("matching>>>"+templateStr);
    Matcher m = g.matcher("format", templateStr);
    int pos = 0;
    while (m.find(pos))
    {
      //m.dumpMatches();
      w.write(templateStr.substring(pos, m.getBegin()));
      String methodName = m.group(0).substring();
      try
      {        
        Object[] args = new Object[m.groupCount()-1];
        for(int i=1;i<m.groupCount();i++)
          args[i-1] = m.group(i).group(0).substring().replace("\\n","\n").replace("\\\"","\"").replace("\\\\","\\");
        
        Method method = methods.get(methodName+"%"+args.length);
        if(method == null)
          throw new NoSuchMethodException(methodName+"%"+args.length+" in '"+m.substring()+"'");
        
        String out = (String)method.invoke(formatObject, args);
        if(out == null) throw new RuntimeException("null returned by "+methodName);
        out = replaceAll(formatObject, out);
        w.write(out);
      } catch (NoSuchMethodException e)
      {
        System.err.println("Warning: No such macro: "+e.getMessage()+" near line "+m.near().toString().replace("{[]}","input"));
        w.write(m.substring());
      } catch (Exception e)
      {
        throw new RuntimeException("For method " + methodName, e);
      } finally
      {
        pos = m.getEnd();
      }
    }
    w.write(templateStr.substring(pos));
    w.flush();
  }

  public String replaceAll(Object formatObject, String out)
  {
    for(int i=0;i<20;i++) {
      String newOut = outputLiteral(out, formatObject);
      if(newOut.equals(out))
        break;
      out = newOut;
    }
    return out;
  }

  /**
   * A basic test/example
   * 
   * @param args
   * @throws Exception
   */
  public static void main(String[] args)
  {
    
    // Create a formatting object
    Object f = new Object()
    {
      public OutputGen outputgen;
      public String foo()
      {
        return "FOO!";
      }

      public String bar(String sep)
      {
        List<String> ret = new ArrayList<String>();
        ret.add("BAR1");
        ret.add("BAR2");
        return join(sep,ret);
      }
      
      public Map<String,Integer> loopVars = new HashMap<String,Integer>();
      public String loop(String var,String lo,String hi,String body) {
        int lov = Integer.parseInt(lo);
        int hiv = Integer.parseInt(hi);
        int del = lov < hiv ? 1 : -1;
        StringBuilder sb = new StringBuilder();
        for(int i=lov;i!=hiv;i += del) {
          loopVars.put(var,i);
          sb.append(outputgen.replaceAll(this, body));
        }
        loopVars.remove(var);
        return sb.toString();
      }
      public String var(String name)
      {
        if(loopVars.get(name)==null) {
          System.err.println("Warning: invalid loop variable '"+name+"' valid="+loopVars.keySet());
          return "";
        }
        return loopVars.get(name).toString();
      }
    };

    // Create an output generator
    OutputGen og = new OutputGen();

    // Create a template
    og.addLines(new String[]{
        "call with arg(%{foo}_l,%foo) {",
        " %loop(i,1,5) %[print(%var('i'));%{foo} i=%var(i)\n %loop(j,1,3) %[j=[ %var(j),%var(i)] ]% x)\n]%",
        "  a=call(%bar(\",\n\")); %gorp",
        "}"
    });
    
    // Generate output
    System.out.println(og.output("$default", f));
  }

  public void addLines(String[] strings)
  {
    for (String s : strings)
    {
      addLine(s);
    }
    finishLines();
  }

  public void readSource(File templateFile) throws IOException
  {
    FileReader fr = new FileReader(templateFile);
    BufferedReader br = new BufferedReader(fr);
    readSource(br);
  }
  public static String join(String sep,List<String> args) {
    StringBuilder sb = new StringBuilder();
    for(int i=0;i<args.size();i++) {
      if(i>0) sb.append(sep);
      sb.append(args.get(i));
    }
    return sb.toString();
  }
}