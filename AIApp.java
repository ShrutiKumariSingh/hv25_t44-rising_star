import java.io.*;
import java.util.*;

interface SpeechProcessor
{
   
    void listen();
    void respond(String input);
}

interface TaskManager
{
    void setReminder(String Time, String message);
    void sendEmail(String too, String subject);
}

class VirtualAssistant implements SpeechProcessor, TaskManager
{
    String name;
    VirtualAssistant(String name){
        this.name=name;
    }
    public void listen(){
        System.out.println("Alexa is Listening");
    }
    public void respond(String input){
        System.out.println("User says:"+input);
    }
    public void setReminder(String Time, String message){
        System.out.println("Meeting Time:"+Time+" "+message);
    }
    public void sendEmail(String too, String subject){
        System.out.println("Sending email to:"+too+" \nSubject"+subject);
    }
    public void displayInfo()
    {
        System.out.println("Assistant Name:"+name+" Capabilities: Speech Processing ,Task Mgmt");
    }

}

class AIApp
{
    public static void main(String args[])
    {
       
        VirtualAssistant ob = new VirtualAssistant("Alexa");
        Scanner sc=new Scanner(System.in);
        //System.out.println("Enter input:");
        ob.listen();
        ob.respond("What is the weather today?");
        ob.setReminder("6:00 pm","Meeting with project team");
        ob.sendEmail("send2vidyut@gmail.com","Project Update");
        ob.displayInfo();

    }
}