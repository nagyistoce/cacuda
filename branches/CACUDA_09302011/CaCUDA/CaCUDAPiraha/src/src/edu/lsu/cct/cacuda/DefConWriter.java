package edu.lsu.cct.cacuda;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;

public class DefConWriter extends Writer
{
  final Writer w;
  public DefConWriter(Writer w) {
    this.w = w;
  }
  
  @Override
  public void close() throws IOException
  {
    w.close();
  }

  @Override
  public void flush() throws IOException
  {
    w.flush();
  }

  @Override
  public void write(char[] str, int start, int end) throws IOException
  {
    for(int i=start;i<end;i++)
      w(str[i]);
  }
  
  StringBuilder sb = new StringBuilder();
  private void w(char c) throws IOException {
    if(c == '\n') {
      if(sb.length()>0 && sb.charAt(sb.length()-1) == '\\') {
        sb.setLength(sb.length()-1);
        while(sb.length()>0 && " \t\r\b".indexOf(sb.charAt(sb.length()-1)) >= 0) {
          sb.setLength(sb.length()-1);
        }
        while(sb.length() < 79) {
          sb.append(' ');
        }
        sb.append('\\');
      }
    }
    sb.append(c);
    if(c == '\n') {
      w.write(sb.toString());
      sb.setLength(0);
    }
  }
  
  /** Test and demonstration */
  public static void main(String[] args) throws Exception {
    OutputStreamWriter osw = new OutputStreamWriter(System.out);
    DefConWriter dcw = new DefConWriter(osw);
    PrintWriter pw = new PrintWriter(dcw);
    pw.println("      hello\\");
    pw.println("world\\");
    pw.println("  and    \\");
    pw.println("goodbye             \\");
    pw.println("moon");
    pw.flush();
  }
}
